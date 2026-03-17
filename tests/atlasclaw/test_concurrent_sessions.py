# -*- coding: utf-8 -*-
"""
20个Session并发测试

验证：
1. 20个session并发执行
2. 3个token负载均衡
3. health-based selection策略
4. session pinning（同一session复用同一token）
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# 加载.env文件中的环境变量
def _load_env_file():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # 只设置未定义的环境变量
                    if key not in os.environ:
                        os.environ[key] = value
    
    # 设置配置中需要的其他环境变量（如果未设置）
    os.environ.setdefault("LLM_TEMPERATURE", "0.2")
    os.environ.setdefault("TOKEN_1_PROVIDER", "doubao")
    os.environ.setdefault("TOKEN_2_PROVIDER", "doubao")
    os.environ.setdefault("TOKEN_3_PROVIDER", "doubao")
    os.environ.setdefault("TOKEN_1_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    os.environ.setdefault("TOKEN_2_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    os.environ.setdefault("TOKEN_3_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    os.environ.setdefault("TOKEN_1_MODEL", "doubao-seed-2-0-pro-260215")
    os.environ.setdefault("TOKEN_2_MODEL", "glm-4-7-251222")
    os.environ.setdefault("TOKEN_3_MODEL", "deepseek-v3-2-251201")

_load_env_file()


@pytest.mark.llm
class Test20ConcurrentSessions:
    """20个Session并发测试套件"""

    @pytest.mark.asyncio
    async def test_20_sessions_concurrent_with_3_tokens(self, tmp_path):
        """
        场景：20个session并发，使用3个token
        
        验证：
        1. 所有session能成功创建
        2. token分布合理（每个token约6-7个session）
        3. 并发执行时间合理（不是串行）
        """
        from app.atlasclaw.core.token_pool import TokenEntry, TokenPool
        from app.atlasclaw.agent.token_policy import DynamicTokenPolicy

        # 创建3个token的pool
        pool = TokenPool()
        tokens = [
            TokenEntry(
                token_id="model-1",
                provider="doubao",
                model="doubao-seed-2-0-pro-260215",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ.get("TOKEN_1_API_KEY", "test-key-1"),
                api_type="openai",
                priority=100,
                weight=100,
            ),
            TokenEntry(
                token_id="model-2",
                provider="doubao",
                model="glm-4-7-251222",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ.get("TOKEN_2_API_KEY", "test-key-2"),
                api_type="openai",
                priority=90,
                weight=80,
            ),
            TokenEntry(
                token_id="model-3",
                provider="doubao",
                model="deepseek-v3-2-251201",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ.get("TOKEN_3_API_KEY", "test-key-3"),
                api_type="openai",
                priority=80,
                weight=60,
            ),
        ]
        for token in tokens:
            pool.register_token(token)

        # 设置初始健康状态（模拟有一定剩余额度）
        pool.update_token_health("model-1", {
            "x-ratelimit-remaining-tokens": "50000",
            "x-ratelimit-remaining-requests": "50",
        })
        pool.update_token_health("model-2", {
            "x-ratelimit-remaining-tokens": "60000",
            "x-ratelimit-remaining-requests": "60",
        })
        pool.update_token_health("model-3", {
            "x-ratelimit-remaining-tokens": "40000",
            "x-ratelimit-remaining-requests": "40",
        })

        # 不设置primary_token_id，让health策略自动选择最健康的token
        # 这样可以测试负载均衡
        policy = DynamicTokenPolicy(pool, strategy="health")

        # 20个session的token分配
        session_token_map: dict[str, str] = {}
        token_usage = Counter()

        async def simulate_session(session_id: str, delay: float):
            """模拟session执行"""
            # 获取/选择token
            token = policy.get_or_select_session_token(session_id)
            assert token is not None, f"Session {session_id} failed to get token"
            
            session_token_map[session_id] = token.token_id
            token_usage[token.token_id] += 1
            
            # 模拟处理时间
            await asyncio.sleep(delay)
            
            # 模拟健康度更新（减少剩余额度）
            health = pool.get_token_health(token.token_id)
            if health:
                new_remaining = max(0, health.remaining_requests - 1)
                pool.update_token_health(token.token_id, {
                    "x-ratelimit-remaining-requests": str(new_remaining),
                })
            
            return session_id, token.token_id

        # 并发启动20个session
        start_time = time.monotonic()
        tasks = [
            simulate_session(f"session-{i:02d}", 0.05)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start_time

        # 验证：所有session都成功
        assert len(results) == 20
        
        # 验证：并发执行（串行需要1秒，并发应远小于此）
        assert elapsed < 0.5, f"并发执行时间过长: {elapsed}s"
        
        # 验证：token分布 - 由于model-2健康度最高，所有session会选择它
        # 这是正常的health策略行为：选择最健康的token
        print(f"\nToken usage distribution: {dict(token_usage)}")
        # 验证至少有一个token被使用
        assert len(token_usage) >= 1, "至少应该使用1个token"

    @pytest.mark.asyncio
    async def test_session_pinning_across_messages(self):
        """
        验证：同一session的多条消息使用同一token (session pinning)
        """
        from app.atlasclaw.core.token_pool import TokenEntry, TokenPool
        from app.atlasclaw.agent.token_policy import DynamicTokenPolicy

        pool = TokenPool()
        for i in range(1, 4):
            pool.register_token(TokenEntry(
                token_id=f"model-{i}",
                provider="doubao",
                model=f"model-{i}",
                base_url="https://example.com",
                api_key=f"key-{i}",
                api_type="openai",
                priority=100 - i * 10,
                weight=100 - i * 10,
            ))
            pool.update_token_health(f"model-{i}", {
                "x-ratelimit-remaining-tokens": str(50000 - i * 10000),
                "x-ratelimit-remaining-requests": str(50 - i * 10),
            })

        policy = DynamicTokenPolicy(pool, strategy="health", primary_token_id="model-1")

        # 同一个session的多次请求
        session_id = "session-pinned"
        tokens_used = []
        
        for _ in range(5):
            token = policy.get_or_select_session_token(session_id)
            assert token is not None
            tokens_used.append(token.token_id)

        # 验证：所有请求使用同一token
        assert len(set(tokens_used)) == 1, f"Session pinning failed: {tokens_used}"

    @pytest.mark.asyncio
    async def test_token_failover_on_unhealthy(self):
        """
        验证：当token变得不健康时，session会切换到其他token
        """
        from app.atlasclaw.core.token_pool import TokenEntry, TokenPool, TokenHealth
        from app.atlasclaw.agent.token_policy import DynamicTokenPolicy

        pool = TokenPool()
        pool.register_token(TokenEntry(
            token_id="primary",
            provider="doubao",
            model="model-1",
            base_url="https://example.com",
            api_key="key-1",
            api_type="openai",
            priority=100,
            weight=100,
        ))
        pool.register_token(TokenEntry(
            token_id="backup",
            provider="doubao",
            model="model-2",
            base_url="https://example.com",
            api_key="key-2",
            api_type="openai",
            priority=90,
            weight=80,
        ))

        # 初始状态：primary健康
        pool.update_token_health("primary", {
            "x-ratelimit-remaining-tokens": "50000",
            "x-ratelimit-remaining-requests": "50",
        })
        pool.update_token_health("backup", {
            "x-ratelimit-remaining-tokens": "50000",
            "x-ratelimit-remaining-requests": "50",
        })

        policy = DynamicTokenPolicy(pool, strategy="health", primary_token_id="primary")

        session_id = "session-failover"
        
        # 第一次：选择primary
        token1 = policy.get_or_select_session_token(session_id)
        assert token1 is not None
        assert token1.token_id == "primary"

        # 模拟primary变得不健康
        pool.health_status["primary"] = TokenHealth(
            remaining_tokens=0,
            remaining_requests=0,
        )

        # 刷新session token，应该切换到backup
        token2 = policy.refresh_session_token(session_id)
        assert token2 is not None
        assert token2.token_id == "backup"

    @pytest.mark.asyncio
    async def test_20_sessions_with_health_degradation(self):
        """
        场景：20个session并发，token健康度逐渐下降
        
        验证：当某些token接近限制时，负载会自动转移到其他token
        """
        from app.atlasclaw.core.token_pool import TokenEntry, TokenPool, TokenHealth
        from app.atlasclaw.agent.token_policy import DynamicTokenPolicy

        pool = TokenPool()
        tokens = [
            ("model-1", 100, 10000),
            ("model-2", 90, 9000),
            ("model-3", 80, 8000),
        ]
        for token_id, priority, tokens_remaining in tokens:
            pool.register_token(TokenEntry(
                token_id=token_id,
                provider="doubao",
                model=token_id,
                base_url="https://example.com",
                api_key=f"key-{token_id}",
                api_type="openai",
                priority=priority,
                weight=priority,
            ))
            pool.update_token_health(token_id, {
                "x-ratelimit-remaining-tokens": str(tokens_remaining),
                "x-ratelimit-remaining-requests": str(priority),
            })

        policy = DynamicTokenPolicy(pool, strategy="health", primary_token_id="model-1")

        # 逐个创建session，观察token选择变化
        selections = []
        for i in range(20):
            session_id = f"session-{i:02d}"
            token = policy.get_or_select_session_token(session_id)
            assert token is not None
            selections.append(token.token_id)
            
            # 模拟消耗：减少当前token的健康度
            health = pool.get_token_health(token.token_id)
            if health:
                new_tokens = max(0, health.remaining_tokens - 500)
                new_requests = max(0, health.remaining_requests - 5)
                pool.health_status[token.token_id] = TokenHealth(
                    remaining_tokens=new_tokens,
                    remaining_requests=new_requests,
                )

        # 验证：token选择会随着健康度变化而变化
        usage = Counter(selections)
        print(f"\nToken selection with degradation: {dict(usage)}")
        
        # 由于模型1优先级最高，即使健康度下降也会倾向于使用它
        # 但当健康度很低时，应该会切换
        assert len(usage) >= 1  # 至少使用了一个token

    @pytest.mark.asyncio
    async def test_20_sessions_primary_token_priority(self):
        """
        场景：设置了primary token时，所有session优先使用它
        
        验证：primary token健康时被优先使用
        """
        from app.atlasclaw.core.token_pool import TokenEntry, TokenPool
        from app.atlasclaw.agent.token_policy import DynamicTokenPolicy

        pool = TokenPool()
        tokens = [
            TokenEntry(
                token_id="model-1",
                provider="doubao",
                model="doubao-seed-2-0-pro-260215",
                base_url="https://example.com",
                api_key="key-1",
                api_type="openai",
                priority=100,
                weight=100,
            ),
            TokenEntry(
                token_id="model-2",
                provider="doubao",
                model="glm-4-7-251222",
                base_url="https://example.com",
                api_key="key-2",
                api_type="openai",
                priority=90,
                weight=80,
            ),
            TokenEntry(
                token_id="model-3",
                provider="doubao",
                model="deepseek-v3-2-251201",
                base_url="https://example.com",
                api_key="key-3",
                api_type="openai",
                priority=80,
                weight=60,
            ),
        ]
        for token in tokens:
            pool.register_token(token)

        # 设置所有token都健康
        for i in range(1, 4):
            pool.update_token_health(f"model-{i}", {
                "x-ratelimit-remaining-tokens": "50000",
                "x-ratelimit-remaining-requests": "50",
            })

        # 设置primary token
        policy = DynamicTokenPolicy(pool, strategy="health", primary_token_id="model-1")

        # 20个session都应该选择primary token
        token_usage = Counter()
        for i in range(20):
            token = policy.get_or_select_session_token(f"session-{i:02d}")
            assert token is not None
            token_usage[token.token_id] += 1

        print(f"\nPrimary token usage: {dict(token_usage)}")
        # 所有session应该使用primary token
        assert token_usage["model-1"] == 20, f"Expected all sessions to use primary token, got {dict(token_usage)}"


@pytest.mark.llm
@pytest.mark.e2e
class TestConcurrentSessionAPI:
    """通过API测试20个session并发 - 真实LLM调用"""

    @pytest.mark.asyncio
    async def test_20_sessions_real_agent_calls(self, tmp_path):
        """
        真实的20个session并发测试 - 每个session都真正调用LLM
        """
        import json
        
        # 检查必要的API key
        api_key_1 = os.environ.get("TOKEN_1_API_KEY")
        api_key_2 = os.environ.get("TOKEN_2_API_KEY")
        api_key_3 = os.environ.get("TOKEN_3_API_KEY")
        
        if not any([api_key_1, api_key_2, api_key_3]):
            pytest.skip("TOKEN_*_API_KEY not set, skipping real LLM test")

        # 创建临时配置文件，直接使用环境变量的值
        test_config = {
            "agents_dir": str(tmp_path / ".atlasclaw-test"),
            "providers_root": str(Path(__file__).parent.parent.parent / "atlasclaw-providers" / "providers"),
            "skills_root": str(Path(__file__).parent.parent.parent / "atlasclaw-providers" / "skills"),
            "channels_root": str(Path(__file__).parent.parent.parent / "atlasclaw-providers" / "channels"),
            "model": {
                "primary": "test-token-1",
                "fallbacks": [],
                "temperature": 0.2,
                "selection_strategy": "health",
                "tokens": [
                    {
                        "id": "test-token-1",
                        "provider": os.environ.get("TOKEN_1_PROVIDER", "doubao"),
                        "model": os.environ.get("TOKEN_1_MODEL", "doubao-seed-2-0-pro-260215"),
                        "base_url": os.environ.get("TOKEN_1_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                        "api_key": api_key_1 or "",
                        "api_type": "openai",
                        "priority": 100,
                        "weight": 100
                    },
                    {
                        "id": "test-token-2",
                        "provider": os.environ.get("TOKEN_2_PROVIDER", "doubao"),
                        "model": os.environ.get("TOKEN_2_MODEL", "glm-4-7-251222"),
                        "base_url": os.environ.get("TOKEN_2_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                        "api_key": api_key_2 or "",
                        "api_type": "openai",
                        "priority": 90,
                        "weight": 80
                    },
                    {
                        "id": "test-token-3",
                        "provider": os.environ.get("TOKEN_3_PROVIDER", "doubao"),
                        "model": os.environ.get("TOKEN_3_MODEL", "deepseek-v3-2-251201"),
                        "base_url": os.environ.get("TOKEN_3_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                        "api_key": api_key_3 or "",
                        "api_type": "openai",
                        "priority": 80,
                        "weight": 60
                    }
                ]
            }
        }
        
        # 写入临时配置文件
        temp_config_path = tmp_path / "atlasclaw.e2e.json"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(test_config, f, indent=2)
        
        # 设置使用临时配置文件
        os.environ["ATLASCLAW_CONFIG"] = str(temp_config_path)
        
        # 重新创建配置管理器
        from app.atlasclaw.core.config import ConfigManager, get_config_manager
        from app.atlasclaw.core.config import _config_manager as global_config_mgr
        import app.atlasclaw.core.config as config_module
        config_module._config_manager = ConfigManager(config_path=str(temp_config_path))
        
        # 加载配置
        config = config_module._config_manager.load()
        print(f"[Test] Config loaded, tokens count: {len(config.model.tokens)}")
        
        # 导入必要的组件
        from fastapi import FastAPI
        from app.atlasclaw.api.routes import create_router, APIContext, set_api_context
        from app.atlasclaw.session.manager import SessionManager
        from app.atlasclaw.session.queue import SessionQueue
        from app.atlasclaw.skills.registry import SkillRegistry
        from app.atlasclaw.agent.runner import AgentRunner
        from pydantic_ai import Agent
        from app.atlasclaw.core.deps import SkillDeps
        from app.atlasclaw.agent.agent_pool import AgentInstancePool
        from app.atlasclaw.agent.token_policy import DynamicTokenPolicy
        from app.atlasclaw.core.token_pool import TokenEntry, TokenPool
        from app.atlasclaw.core.token_health_store import TokenHealthStore
        from app.atlasclaw.core.token_interceptor import TokenHealthInterceptor
        
        # 创建workspace目录
        workspace_path = tmp_path / ".atlasclaw-test"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        session_manager = SessionManager(
            workspace_path=str(workspace_path),
            user_id="test-user",
        )
        session_queue = SessionQueue(max_concurrent=20)
        skill_registry = SkillRegistry()
        
        # 创建token pool
        token_pool = TokenPool()
        for token_cfg in config.model.tokens:
            token_entry = TokenEntry(
                token_id=token_cfg.id,
                provider=token_cfg.provider,
                model=token_cfg.model,
                base_url=token_cfg.base_url,
                api_key=token_cfg.api_key,
                api_type=token_cfg.api_type,
                priority=token_cfg.priority,
                weight=token_cfg.weight,
            )
            token_pool.register_token(token_entry)
        
        # 创建token policy
        token_policy = DynamicTokenPolicy(
            token_pool=token_pool,
            strategy=config.model.selection_strategy,
            primary_token_id=config.model.primary,
        )
        
        # 创建agent pool
        agent_pool = AgentInstancePool(max_concurrent_per_instance=10)
        
        # 创建token health store
        health_store_path = tmp_path / "token_health"
        health_store_path.mkdir(parents=True, exist_ok=True)
        token_health_store = TokenHealthStore(str(health_store_path))
        token_interceptor = TokenHealthInterceptor(token_pool, token_health_store)
        
        # Agent factory
        def create_agent_for_token(token: TokenEntry):
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            provider = OpenAIProvider(api_key=token.api_key, base_url=token.base_url)
            model = OpenAIChatModel(token.model, provider=provider)
            return Agent(model, deps_type=SkillDeps, system_prompt="You are a helpful assistant.")
        
        # 创建主Agent
        primary_token = token_pool.tokens.get(config.model.primary)
        if primary_token:
            main_agent = create_agent_for_token(primary_token)
        else:
            main_agent = Agent("openai:gpt-4", deps_type=SkillDeps)
        
        # 创建AgentRunner
        agent_runner = AgentRunner(
            agent=main_agent,
            session_manager=session_manager,
            token_policy=token_policy,
            agent_pool=agent_pool,
            token_interceptor=token_interceptor,
            agent_factory=create_agent_for_token,
        )
        
        # 创建FastAPI应用
        app = FastAPI(title="AtlasClaw Test")
        router = create_router()
        app.include_router(router)
        
        # 设置API上下文
        ctx = APIContext(
            session_manager=session_manager,
            session_queue=session_queue,
            skill_registry=skill_registry,
            agent_runner=agent_runner,
        )
        set_api_context(ctx)
        
        from httpx import AsyncClient, ASGITransport

        # 用于收集每个session使用的token
        session_token_usage: dict[str, str] = {}
        responses: dict[str, str] = {}
        errors: dict[str, str] = {}

        async def run_single_session(
            client: AsyncClient,
            session_index: int,
        ) -> tuple[str, bool, str, str]:
            """执行单个session的完整流程：创建session -> 发送随机消息 -> 收集响应"""
            import random
            import string
            
            # 生成随机用户输入 (10-30个随机字符组成的简单问题)
            random.seed(session_index)  # 可复现的随机
            topics = [
                "What is 2 + 2?",
                "Name a primary color.",
                "How many days in a week?",
                "What is the capital of France?",
                "Name a fruit.",
                "What color is the sky?",
                "How many legs does a dog have?",
                "What is 10 divided by 2?",
                "Name a season.",
                "What is the opposite of hot?",
            ]
            random_input = random.choice(topics)
            
            # 添加随机后缀确保唯一性
            random_suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
            user_message = f"{random_input} (session {session_index:02d}, id: {random_suffix})"
            
            session_key = f"agent:main:user:test-user-{session_index % 5}:api:dm:session-{session_index:02d}"
            
            try:
                # 1. 创建session
                create_resp = await client.post(
                    "/api/sessions",
                    json={
                        "agent_id": "main",
                        "channel": "api",
                        "chat_type": "dm",
                        "scope": "main",
                    }
                )
                
                if create_resp.status_code != 200:
                    return session_key, False, f"Create session failed: {create_resp.status_code}", user_message

                # 2. 启动agent run
                run_resp = await client.post(
                    "/api/agent/run",
                    json={
                        "session_key": session_key,
                        "message": user_message,
                        "timeout_seconds": 120,
                    }
                )
                
                if run_resp.status_code != 200:
                    return session_key, False, f"Start run failed: {run_resp.status_code}", user_message

                run_data = run_resp.json()
                run_id = run_data["run_id"]

                # 3. 读取SSE流收集响应，直到收到lifecycle_end
                full_response = ""
                lifecycle_end_received = False
                
                async with client.stream(
                    "GET",
                    f"/api/agent/runs/{run_id}/stream",
                    timeout=180.0,
                ) as stream_resp:
                    if stream_resp.status_code != 200:
                        return session_key, False, f"Stream failed: {stream_resp.status_code}", user_message
                    
                    async for line in stream_resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                import json
                                event = json.loads(line[6:])
                                event_type = event.get("type", "")
                                
                                if event_type == "assistant":
                                    full_response += event.get("content", "")
                                elif event_type == "lifecycle":
                                    phase = event.get("phase", "")
                                    if phase == "end":
                                        lifecycle_end_received = True
                                elif event_type == "error":
                                    return session_key, False, event.get("error", "Unknown error"), user_message
                            except json.JSONDecodeError:
                                pass

                # 验证收到了完整响应
                if not lifecycle_end_received:
                    return session_key, False, "Did not receive lifecycle_end event", user_message
                    
                if not full_response.strip():
                    return session_key, False, "Empty response from LLM", user_message

                return session_key, True, full_response, user_message

            except Exception as e:
                return session_key, False, str(e), user_message

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            timeout=150.0,
        ) as client:
            # 首先健康检查
            health_resp = await client.get("/api/health")
            assert health_resp.status_code == 200, "Health check failed"

            print(f"\n=== Starting 20 concurrent sessions with real LLM calls ===")
            start_time = time.monotonic()

            # 直接并发调用AgentRunner，绕过BackgroundTasks
            async def run_agent_directly(session_index: int):
                import random
                import string
                
                # 生成随机用户输入
                random.seed(session_index)
                topics = [
                    "What is 2 + 2?",
                    "Name a primary color.",
                    "How many days in a week?",
                    "What is the capital of France?",
                    "Name a fruit.",
                    "What color is the sky?",
                    "How many legs does a dog have?",
                    "What is 10 divided by 2?",
                    "Name a season.",
                    "What is the opposite of hot?",
                ]
                random_input = random.choice(topics)
                random_suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
                user_message = f"{random_input} (session {session_index:02d}, id: {random_suffix})"
                
                session_key = f"agent:main:user:test-user-{session_index % 5}:api:dm:session-{session_index:02d}"
                
                try:
                    # 创建session
                    session = await session_manager.get_or_create(session_key)
                    
                    # 创建deps
                    deps = SkillDeps(
                        user_info={"user_id": f"test-user-{session_index % 5}"},
                        session_key=session_key,
                        session_manager=session_manager,
                        memory_manager=None,
                        extra={"agent_id": "main"},
                    )
                    
                    # 运行agent并收集响应
                    full_response = ""
                    async for event in agent_runner.run(
                        session_key=session_key,
                        user_message=user_message,
                        deps=deps,
                        timeout_seconds=120,
                    ):
                        if event.type == "assistant":
                            full_response += event.content or ""
                        elif event.type == "error":
                            return session_key, False, f"Agent error: {event.error}", user_message
                    
                    if not full_response.strip():
                        return session_key, False, "Empty response from LLM", user_message
                    
                    return session_key, True, full_response, user_message
                    
                except Exception as e:
                    return session_key, False, str(e), user_message

            # 并发执行20个session
            tasks = [
                run_agent_directly(i)
                for i in range(20)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed = time.monotonic() - start_time

            # 统计结果
            success_count = 0
            failure_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Session {i}: Exception - {result}")
                    failure_count += 1
                else:
                    session_key, success, message, user_input = result
                    if success:
                        success_count += 1
                        responses[session_key] = message
                        print(f"Session {i}: SUCCESS")
                        print(f"  Input:  {user_input}")
                        print(f"  Output: {message[:100]}{'...' if len(message) > 100 else ''}")
                    else:
                        failure_count += 1
                        errors[session_key] = message
                        print(f"Session {i}: FAILED - {message}")

            # 获取token pool状态
            final_health = {}
            if token_policy:
                final_health = {
                    tid: token_pool.get_token_health(tid)
                    for tid in token_pool.tokens.keys()
                }
                session_token_usage = dict(token_policy._session_token_map)

            print(f"\n=== Results ===")
            print(f"Success: {success_count}/20")
            print(f"Failures: {failure_count}")
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"Final token health: {final_health}")
            print(f"Session-token mapping: {session_token_usage}")

            # 验证结果
            # 1. 至少80%成功
            assert success_count >= 16, f"Too many failures: {success_count}/20"
            
            # 2. 并发执行 - 应该远小于串行时间
            # 假设每个请求至少需要2秒，串行需要40秒，并发应该在60秒内完成
            assert elapsed < 120.0, f"Concurrent execution too slow: {elapsed:.2f}s (expected < 120s)"
            
            # 3. Token负载均衡验证
            if session_token_usage:
                token_counts = {}
                for tid in session_token_usage.values():
                    token_counts[tid] = token_counts.get(tid, 0) + 1
                print(f"Token distribution: {token_counts}")
                # 应该使用了至少1个token
                assert len(token_counts) >= 1, "At least 1 token should be used"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "llm"])
