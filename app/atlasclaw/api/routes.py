# -*- coding: utf-8 -*-
"""


REST API

implementsession management, Agent run, Skills, etc. REST.
corresponds to tasks.md 7.2.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Header, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..session.manager import SessionManager
from ..session.context import SessionKey, SessionScope, ChatType
from ..session.queue import SessionQueue, QueueMode
from ..skills.registry import SkillRegistry
from ..memory.manager import MemoryManager
from ..core.deps import SkillDeps
from ..auth.models import UserInfo, ANONYMOUS_USER
from .sse import SSEManager, SSEEvent, SSEEventType
from .webhook_dispatch import (
    WebhookDispatchManager,
    WebhookSystemIdentity,
    build_webhook_user_message,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic / model
# ============================================================================

class SessionCreateRequest(BaseModel):
    """createsession"""
    agent_id: str = "main"
    channel: str = "api"
    chat_type: str = "dm"
    scope: str = "main"


class SessionResponse(BaseModel):
    """session"""
    session_key: str
    agent_id: str
    channel: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    total_tokens: int


class SessionResetRequest(BaseModel):
    """Reset a session"""
    archive: bool = True


class AgentRunRequest(BaseModel):
    """Agent run"""
    session_key: str
    message: str
    model: Optional[str] = None
    timeout_seconds: int = 600


class AgentRunResponse(BaseModel):
    """Agent run"""
    run_id: str
    status: str
    session_key: str


class AgentStatusResponse(BaseModel):
    """Agent"""
    run_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tokens_used: int = 0
    error: Optional[str] = None


class SkillExecuteRequest(BaseModel):
    """Skill execute"""
    skill_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class SkillExecuteResponse(BaseModel):
    """Skill execute"""
    skill_name: str
    result: Any
    duration_ms: int


class MemorySearchRequest(BaseModel):
    """search"""
    query: str
    top_k: int = 10
    apply_recency: bool = True


class MemorySearchResult(BaseModel):
    """Search results"""
    id: str
    content: str
    score: float
    source: str
    timestamp: datetime
    highlights: list[str]


class MemoryWriteRequest(BaseModel):
    """"""
    content: str
    memory_type: str = "daily"  # daily / long_term
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    section: str = "General"


class QueueModeRequest(BaseModel):
    """Queue mode"""
    mode: str  # collect / steer / followup / steer-backlog / interrupt


class StatusResponse(BaseModel):
    """"""
    session_key: str
    context_tokens: int
    input_tokens: int
    output_tokens: int
    queue_mode: str
    queue_size: int


class CompactRequest(BaseModel):
    """"""
    instruction: Optional[str] = None


class WebhookDispatchRequest(BaseModel):
    """Webhook markdown-skill dispatch request."""
    skill: str
    args: dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    timeout_seconds: int = 600


class WebhookDispatchResponse(BaseModel):
    """Webhook acknowledgement payload."""
    status: str


# ============================================================================
# API context
# ============================================================================

@dataclass
class APIContext:
    """

API context
 
 contains inject.
 
"""
    session_manager: SessionManager
    session_queue: SessionQueue
    skill_registry: SkillRegistry
    memory_manager: Optional[MemoryManager] = None
    sse_manager: Optional[SSEManager] = None
    agent_runner: Optional[Any] = None  # AgentRunner instance
    service_provider_registry: Optional[Any] = None  # ServiceProviderRegistry instance
    available_providers: dict[str, list[str]] = None
    provider_instances: dict[str, dict[str, dict[str, Any]]] = None
    webhook_manager: Optional[WebhookDispatchManager] = None
    
    # run
    active_runs: dict[str, dict[str, Any]] = None
    
    def __post_init__(self):
        if self.active_runs is None:
            self.active_runs = {}
        if self.sse_manager is None:
            self.sse_manager = SSEManager()
        if self.available_providers is None:
            self.available_providers = {}
        if self.provider_instances is None:
            self.provider_instances = {}


# context(apply)
_api_context: Optional[APIContext] = None


def set_api_context(ctx: APIContext) -> None:
    """API context"""
    global _api_context
    _api_context = ctx


def get_api_context() -> APIContext:
    """get API context"""
    if _api_context is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API context not initialized"
        )
    return _api_context


def _safe_decode_request_body(body: bytes, max_chars: int = 1000) -> str:
    if not body:
        return "<empty>"

    try:
        parsed = json.loads(body)
        text = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        text = body.decode("utf-8", errors="replace")

    if len(text) > max_chars:
        return f"{text[:max_chars]}...<truncated>"
    return text


def install_request_validation_logging(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        body = await request.body()
        logger.warning(
            "Request validation failed: method=%s path=%s errors=%s body=%s",
            request.method,
            request.url.path,
            exc.errors(),
            _safe_decode_request_body(body),
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _build_scoped_deps(
    ctx: APIContext,
    user_info: UserInfo,
    session_key: str,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> SkillDeps:
    """Create request-scoped dependencies for agent-style execution."""
    # Check if session manager is in legacy mode
    if getattr(ctx.session_manager, '_legacy_mode', False):
        scoped_session_mgr = SessionManager(
            agents_dir=str(ctx.session_manager.agents_dir),
            agent_id=ctx.session_manager.agent_id,
            user_id=user_info.user_id,
        )
    else:
        # Use workspace-based session manager
        scoped_session_mgr = SessionManager(
            workspace_path=str(ctx.session_manager.workspace_path),
            user_id=user_info.user_id,
        )
    scoped_memory_mgr: Optional[MemoryManager] = None
    if ctx.memory_manager is not None:
        scoped_memory_mgr = MemoryManager(
            workspace=str(ctx.memory_manager._workspace),
            user_id=user_info.user_id,
        )

    deps_extra = {
        "_service_provider_registry": ctx.service_provider_registry,
        "available_providers": ctx.available_providers,
        "provider_instances": ctx.provider_instances,
        "skills_snapshot": ctx.skill_registry.snapshot(),
        "md_skills_snapshot": ctx.skill_registry.md_snapshot(),
    }
    if extra:
        deps_extra.update(extra)

    return SkillDeps(
        user_info=user_info,
        session_key=session_key,
        session_manager=scoped_session_mgr,
        memory_manager=scoped_memory_mgr,
        extra=deps_extra,
    )


# ============================================================================
# Agent Execution Helper Functions
# ============================================================================

async def _execute_agent_run(
    ctx: APIContext,
    run_id: str,
    session_key: str,
    message: str,
    timeout_seconds: int,
    user_info: Optional[UserInfo] = None,
) -> None:
    """
    Execute Agent run in background and push events via SSE
    
    Args:
        ctx: API context
        run_id: Run ID
        session_key: Session key
        message: User message
        timeout_seconds: Timeout in seconds
        user_info: Authenticated user identity (injected by AuthMiddleware)
    """
    import asyncio
    
    _user_info = user_info or ANONYMOUS_USER
    
    try:
        # AgentRunner must be configured
        if not ctx.agent_runner:
            raise RuntimeError(
                "AgentRunner not configured. "
                "Ensure LLM provider is properly configured in atlasclaw.json"
            )
        
        deps = _build_scoped_deps(ctx, _user_info, session_key)

        async for event in ctx.agent_runner.run(
            session_key=session_key,
            user_message=message,
            deps=deps,
            timeout_seconds=timeout_seconds
        ):
            # Convert StreamEvent to SSE event
            if event.type == "lifecycle":
                ctx.sse_manager.push_lifecycle(run_id, event.phase)
            elif event.type == "assistant":
                ctx.sse_manager.push_assistant(run_id, event.content)
            elif event.type == "tool":
                result_str = str(event.content) if event.content else None
                ctx.sse_manager.push_tool(
                    run_id, 
                    event.tool, 
                    event.phase,
                    result=result_str
                )
            elif event.type == "error":
                ctx.sse_manager.push_error(run_id, event.error)
        
        # Update run status
        if run_id in ctx.active_runs:
            ctx.active_runs[run_id]["status"] = "completed"
            ctx.active_runs[run_id]["completed_at"] = datetime.now(timezone.utc)
        
    except asyncio.TimeoutError:
        ctx.sse_manager.push_error(run_id, "Agent execution timed out")
        ctx.sse_manager.push_lifecycle(run_id, "error")
        if run_id in ctx.active_runs:
            ctx.active_runs[run_id]["status"] = "timeout"
            ctx.active_runs[run_id]["error"] = "Execution timed out"
            
    except Exception as e:
        error_msg = str(e)
        ctx.sse_manager.push_error(run_id, error_msg)
        ctx.sse_manager.push_lifecycle(run_id, "error")
        if run_id in ctx.active_runs:
            ctx.active_runs[run_id]["status"] = "error"
            ctx.active_runs[run_id]["error"] = error_msg
            
    finally:
        # Close SSE stream
        ctx.sse_manager.close_stream(run_id)


async def _execute_webhook_dispatch(
    ctx: APIContext,
    dispatch_id: str,
    system: WebhookSystemIdentity,
    skill_entry: Any,
    session_key: str,
    agent_id: str,
    args: dict[str, Any],
    timeout_seconds: int,
) -> None:
    """Execute a webhook-triggered markdown skill without exposing a result stream."""
    if not ctx.agent_runner:
        logger.error("Webhook dispatch %s failed: AgentRunner not configured", dispatch_id)
        return

    user_info = UserInfo(
        user_id=f"webhook-{system.system_id}",
        display_name=system.system_id,
        roles=["webhook"],
        extra={"system_id": system.system_id},
    )
    user_message = build_webhook_user_message(skill_entry, args, system.system_id)
    deps = _build_scoped_deps(
        ctx,
        user_info,
        session_key,
        extra={
            "webhook_skill": skill_entry.qualified_name,
            "webhook_args": dict(args),
            "target_md_skill": {
                "name": skill_entry.name,
                "provider": skill_entry.provider,
                "qualified_name": skill_entry.qualified_name,
                "file_path": skill_entry.file_path,
            },
        },
    )

    logger.info(
        "Accepted webhook dispatch: dispatch_id=%s system_id=%s agent_id=%s skill=%s",
        dispatch_id,
        system.system_id,
        agent_id,
        skill_entry.qualified_name,
    )
    try:
        async for _event in ctx.agent_runner.run(
            session_key=session_key,
            user_message=user_message,
            deps=deps,
            timeout_seconds=timeout_seconds,
        ):
            pass
        logger.info(
            "Webhook dispatch completed: dispatch_id=%s system_id=%s skill=%s",
            dispatch_id,
            system.system_id,
            skill_entry.qualified_name,
        )
    except Exception:
        logger.exception(
            "Webhook dispatch failed: dispatch_id=%s system_id=%s skill=%s",
            dispatch_id,
            system.system_id,
            skill_entry.qualified_name,
        )


# ============================================================================
# create
# ============================================================================

def create_router() -> APIRouter:
    """create API"""
    router = APIRouter(prefix="/api", tags=["AtlasClaw API"])
    
    # ----- session management API -----
    
    @router.post("/sessions", response_model=SessionResponse)
    async def create_session(
        request_obj: Request,
        request: SessionCreateRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> SessionResponse:
        """Create a new session"""
        # Derive user identity from the AuthMiddleware-injected UserInfo
        auth_user: UserInfo = getattr(request_obj.state, "user_info", ANONYMOUS_USER)
        
        key = SessionKey(
            agent_id=request.agent_id,
            channel=request.channel,
            chat_type=ChatType(request.chat_type),
            user_id=auth_user.user_id,
        )
        session_key_str = key.to_string(scope=SessionScope(request.scope))
        
        session = await ctx.session_manager.get_or_create(session_key_str)
        
        return SessionResponse(
            session_key=session_key_str,
            agent_id=key.agent_id,
            channel=key.channel,
            user_id=key.user_id,
            created_at=session.created_at,
            last_activity=session.updated_at,
            message_count=getattr(session, "message_count", 0),
            total_tokens=session.total_tokens
        )
        
    @router.get("/sessions/{session_key}", response_model=SessionResponse)
    async def get_session(
        session_key: str,
        ctx: APIContext = Depends(get_api_context)
    ) -> SessionResponse:
        """get session"""
        session = await ctx.session_manager.get_session(session_key)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_key}"
            )
            
        key = SessionKey.from_string(session_key)
        
        return SessionResponse(
            session_key=session_key,
            agent_id=key.agent_id,
            channel=key.channel,
            user_id=key.user_id,
            created_at=session.created_at,
            last_activity=session.updated_at,
            message_count=getattr(session, "message_count", 0),
            total_tokens=session.total_tokens
        )
        
    @router.post("/sessions/{session_key}/reset")
    async def reset_session(
        session_key: str,
        request: SessionResetRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """Reset a session"""
        await ctx.session_manager.reset_session(session_key, archive=request.archive)
            
        return {"status": "reset", "session_key": session_key}
        
    @router.delete("/sessions/{session_key}")
    async def delete_session(
        session_key: str,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """Delete a session"""
        success = await ctx.session_manager.delete_session(session_key)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_key}"
            )
            
        return {"status": "deleted", "session_key": session_key}
        
    # ----- Agent run API -----
    
    @router.post("/agent/run", response_model=AgentRunResponse)
    async def start_agent_run(
        request_obj: Request,
        request: AgentRunRequest,
        background_tasks: "BackgroundTasks",
        ctx: APIContext = Depends(get_api_context)
    ) -> AgentRunResponse:
        """Agent run"""
        run_id = str(uuid.uuid4())
        
        # Extract UserInfo injected by AuthMiddleware
        user_info: UserInfo = getattr(request_obj.state, "user_info", ANONYMOUS_USER)
        logger.info(
            "Accepted agent run: run_id=%s session_key=%s user_id=%s timeout_seconds=%s message_length=%s",
            run_id,
            request.session_key,
            user_info.user_id,
            request.timeout_seconds,
            len(request.message),
        )
        
        # run
        ctx.active_runs[run_id] = {
            "status": "running",
            "session_key": request.session_key,
            "started_at": datetime.now(timezone.utc),
            "message": request.message,
            "timeout_seconds": request.timeout_seconds
        }
        
        # create SSE stream
        ctx.sse_manager.create_stream(run_id)
        
        # run Agent in background
        background_tasks.add_task(
            _execute_agent_run,
            ctx,
            run_id,
            request.session_key,
            request.message,
            request.timeout_seconds,
            user_info,
        )
        
        return AgentRunResponse(
            run_id=run_id,
            status="running",
            session_key=request.session_key
        )
    
    @router.get("/agent/runs/{run_id}/stream")
    async def stream_agent_run(
        run_id: str,
        last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
        ctx: APIContext = Depends(get_api_context)
    ):
        """
        SSE streaming endpoint
        
        Returns streaming events for Agent run:
        - lifecycle: start/end events
        - assistant: assistant response content
        - tool: tool execution events
        - error: error events
        """
        if run_id not in ctx.active_runs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run not found: {run_id}"
            )
        
        return await ctx.sse_manager.create_response(
            run_id,
            last_event_id=last_event_id
        )
        
    @router.get("/agent/runs/{run_id}", response_model=AgentStatusResponse)
    async def get_agent_status(
        run_id: str,
        ctx: APIContext = Depends(get_api_context)
    ) -> AgentStatusResponse:
        """get Agent run"""
        run_info = ctx.active_runs.get(run_id)
        
        if not run_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run not found: {run_id}"
            )
            
        return AgentStatusResponse(
            run_id=run_id,
            status=run_info.get("status", "unknown"),
            started_at=run_info.get("started_at"),
            completed_at=run_info.get("completed_at"),
            tokens_used=run_info.get("tokens_used", 0),
            error=run_info.get("error")
        )
        
    @router.post("/agent/runs/{run_id}/abort")
    async def abort_agent_run(
        run_id: str,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """in Agent run"""
        run_info = ctx.active_runs.get(run_id)
        
        if not run_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run not found: {run_id}"
            )
            
        run_info["status"] = "aborted"
        # in through abort_signal implement
        
        return {"status": "aborted", "run_id": run_id}
        
    # ----- Skills API -----
    
    @router.get("/skills")
    async def list_skills(
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """available Skills"""
        # Get executable skills (Python handlers)
        executable_skills = ctx.skill_registry.snapshot()
        # Get markdown skills
        md_skills = ctx.skill_registry.md_snapshot()
        
        # Combine both types
        all_skills = []
        for s in executable_skills:
            all_skills.append({
                "name": s["name"],
                "description": s["description"],
                "category": s.get("category", "utility"),
                "type": "executable"
            })
        for s in md_skills:
            # Use qualified_name if available (includes provider prefix)
            name = s.get("qualified_name") or s["name"]
            all_skills.append({
                "name": name,
                "description": s["description"],
                "category": s.get("metadata", {}).get("category", "skill"),
                "type": "markdown"
            })
        
        return {"skills": all_skills}
        
    @router.post("/skills/execute", response_model=SkillExecuteResponse)
    async def execute_skill(
        request: SkillExecuteRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> SkillExecuteResponse:
        """execute Skill"""
        import time
        start = time.monotonic()
        
        try:
            result = await ctx.skill_registry.execute(
                request.skill_name,
                json.dumps(request.args),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Skill execution failed: {str(e)}"
            )
            
        duration_ms = int((time.monotonic() - start) * 1000)
        
        return SkillExecuteResponse(
            skill_name=request.skill_name,
            result=result,
            duration_ms=duration_ms
        )

    @router.post(
        "/webhook/dispatch",
        response_model=WebhookDispatchResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def dispatch_webhook_skill(
        request_obj: Request,
        request: WebhookDispatchRequest,
        background_tasks: "BackgroundTasks",
        ctx: APIContext = Depends(get_api_context),
    ) -> WebhookDispatchResponse:
        """Accept a webhook dispatch for a provider-qualified markdown skill."""
        manager = ctx.webhook_manager
        if manager is None or not manager.enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Webhook dispatch not enabled",
            )

        secret = request_obj.headers.get(manager.header_name, "").strip()
        system = manager.authenticate(secret)
        if system is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook secret",
            )

        try:
            skill_entry = manager.resolve_allowed_skill(system, request.skill)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

        if skill_entry is None:
            if request.skill in ctx.skill_registry.list_skills():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Webhook skill {request.skill!r} resolves to an executable tool, not a markdown skill",
                )
            if request.skill not in system.allowed_skills:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Webhook skill {request.skill!r} is not allowed for system {system.system_id!r}",
                )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook markdown skill not found: {request.skill}",
            )

        agent_id = request.agent_id or system.default_agent_id
        session_key = SessionKey(
            agent_id=agent_id,
            user_id=f"webhook-{system.system_id}",
            channel="webhook",
            chat_type=ChatType.DM,
            peer_id=system.system_id,
        ).to_string(scope=SessionScope.PER_CHANNEL_PEER)

        background_tasks.add_task(
            _execute_webhook_dispatch,
            ctx,
            str(uuid.uuid4()),
            system,
            skill_entry,
            session_key,
            agent_id,
            request.args,
            request.timeout_seconds,
        )
        return WebhookDispatchResponse(status="accepted")
        
    # ----- API -----
    
    @router.post("/memory/search")
    async def search_memory(
        request: MemorySearchRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """search"""
        if not ctx.memory_manager:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Memory system not configured"
            )
            
        # use Hybrid-Searcher
        # implement:return
        return {"results": [], "query": request.query}
        
    @router.post("/memory/write")
    async def write_memory(
        request: MemoryWriteRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """"""
        if not ctx.memory_manager:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Memory system not configured"
            )
            
        if request.memory_type == "daily":
            entry = await ctx.memory_manager.write_daily(
                request.content,
                source=request.source,
                tags=request.tags
            )
        else:
            entry = await ctx.memory_manager.write_long_term(
                request.content,
                source=request.source,
                tags=request.tags,
                section=request.section
            )
            
        return {
            "id": entry.id,
            "memory_type": request.memory_type,
            "timestamp": entry.timestamp.isoformat()
        }
        
    # ----- API -----
    
    @router.get("/sessions/{session_key}/status", response_model=StatusResponse)
    async def get_status(
        session_key: str,
        ctx: APIContext = Depends(get_api_context)
    ) -> StatusResponse:
        """get session"""
        session = await ctx.session_manager.get_session(session_key)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_key}"
            )
            
        queue_mode = ctx.session_queue.get_mode(session_key)
        queue_size = ctx.session_queue.queue_size(session_key)
        
        return StatusResponse(
            session_key=session_key,
            context_tokens=session.context_tokens,
            input_tokens=session.input_tokens,
            output_tokens=session.output_tokens,
            queue_mode=queue_mode.value,
            queue_size=queue_size,
        )
        
    @router.post("/sessions/{session_key}/queue")
    async def set_queue_mode(
        session_key: str,
        request: QueueModeRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """Queue mode"""
        try:
            mode = QueueMode(request.mode)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid queue mode: {request.mode}"
            )
            
        ctx.session_queue.set_session_mode(session_key, mode)
        
        return {"session_key": session_key, "queue_mode": request.mode}
        
    @router.post("/sessions/{session_key}/compact")
    async def trigger_compact(
        session_key: str,
        request: CompactRequest,
        ctx: APIContext = Depends(get_api_context)
    ) -> dict[str, Any]:
        """trigger"""
        session = await ctx.session_manager.get_session(session_key)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_key}"
            )
            
        # at Compaction-Pipeline in
        # return
        return {
            "session_key": session_key,
            "status": "compaction_triggered",
            "instruction": request.instruction
        }
        
    # ----- check -----
    
    @router.get("/health")
    async def health_check() -> dict[str, Any]:
        """check"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    return router
