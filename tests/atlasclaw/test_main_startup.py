# -*- coding: utf-8 -*-
"""
main.py 启动流程测试

测试 FastAPI 应用的 lifespan 初始化流程。
验证所有组件正确初始化：SessionManager, SkillRegistry, AgentRunner 等。
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class TestMainStartup:
    """测试 main.py 启动流程"""

    def test_import_main_module(self):
        """验证可以导入 main 模块"""
        from app.atlasclaw import main
        assert main is not None

    def test_app_instance_exists(self):
        """验证 FastAPI app 实例存在"""
        from app.atlasclaw.main import app
        assert app is not None
        assert "AtlasClaw" in app.title

    def test_app_has_lifespan(self):
        """验证 app 有 lifespan 配置"""
        from app.atlasclaw.main import app
        assert app.router.lifespan_context is not None

    def test_config_loading(self, test_config_path):
        """验证配置文件加载"""
        from app.atlasclaw.core.config import ConfigManager
        
        config_manager = ConfigManager(config_path=str(test_config_path))
        config = config_manager.load()
        assert config is not None
        assert config.model.primary == "test-token-1"
        assert len(config.model.tokens) == 3

    def test_startup_with_env_vars_succeeds(self, test_config_path):
        """验证有环境变量配置时启动成功"""
        import importlib
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        
        # 重新加载模块
        import app.atlasclaw.main as main_module
        importlib.reload(main_module)
        
        # 创建测试客户端应该成功
        with TestClient(main_module.app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"
            from app.atlasclaw.api.routes import get_api_context

            qualified_names = set(get_api_context().skill_registry.list_md_qualified_skills())
            assert "jira:jira-issue" in qualified_names, "should load the external Jira skill as provider-qualified"
            assert "smartcmp:preapproval-agent" in qualified_names, "should load SmartCMP skills as provider-qualified"



class TestConfigResolution:
    """测试配置解析"""

    def test_provider_config_resolution(self, test_config_path):
        """验证 provider 配置解析"""
        from app.atlasclaw.core.config import ConfigManager
        
        config_manager = ConfigManager(config_path=str(test_config_path))
        config = config_manager.load()
        
        # 验证 model 配置 - 现在使用 tokens 配置
        assert config.model.primary == "test-token-1"
        assert len(config.model.tokens) == 3

class TestSimpleLLMCall:
    """简单LLM调用测试 - 验证基础功能"""

    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_simple_agent_call_to_llm(self):
        """
        最简单的LLM调用测试
        
        验证：
        1. Agent能成功调用LLM
        2. 能收到有效的响应
        3. 事件流正常工作
        """
        import os
        import tempfile
        from pathlib import Path
        from dotenv import load_dotenv
        
        # 加载.env文件
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
        
        # 检查环境变量
        token_1_api_key = os.environ.get("TOKEN_1_API_KEY")
        token_1_base_url = os.environ.get("TOKEN_1_BASE_URL")
        token_1_model = os.environ.get("TOKEN_1_MODEL", "deepseek-chat")
        token_1_provider = os.environ.get("TOKEN_1_PROVIDER", "deepseek")
        
        if not token_1_api_key or not token_1_base_url:
            pytest.skip("TOKEN_1_API_KEY and TOKEN_1_BASE_URL must be set")
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_path = Path(tmp_dir) / ".atlasclaw"
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # 创建必要的目录结构
            (workspace_path / "users" / "default" / "sessions").mkdir(parents=True, exist_ok=True)
            (workspace_path / "agents").mkdir(parents=True, exist_ok=True)
            
            # 创建默认agent配置
            agent_config = {
                "id": "main",
                "display_name": "Test Agent",
                "system_prompt": "You are a helpful assistant. Respond briefly."
            }
            import json
            with open(workspace_path / "agents" / "main.json", "w", encoding="utf-8") as f:
                json.dump(agent_config, f)
            
            # 初始化组件
            from app.atlasclaw.session.manager import SessionManager
            from app.atlasclaw.session.queue import SessionQueue
            from app.atlasclaw.skills.registry import SkillRegistry
            from app.atlasclaw.agent.runner import AgentRunner
            from app.atlasclaw.agent.prompt_builder import PromptBuilder, PromptBuilderConfig
            from app.atlasclaw.core.token_pool import TokenPool, TokenEntry
            from app.atlasclaw.agent.token_policy import DynamicTokenPolicy
            from app.atlasclaw.agent.agent_pool import AgentInstancePool
            from app.atlasclaw.core.token_health_store import TokenHealthStore
            from app.atlasclaw.core.token_interceptor import TokenHealthInterceptor
            from app.atlasclaw.core.deps import SkillDeps
            from app.atlasclaw.agent.agent_definition import AgentLoader
            from pydantic_ai import Agent
            
            # 创建TokenPool
            token_pool = TokenPool()
            token_entry = TokenEntry(
                token_id="test-token-1",
                provider=token_1_provider,
                model=token_1_model,
                base_url=token_1_base_url,
                api_key=token_1_api_key,
                api_type="openai",
                priority=100,
                weight=100,
            )
            token_pool.register_token(token_entry)
            
            # 创建组件
            session_manager = SessionManager(
                workspace_path=str(workspace_path),
                user_id="default",
                reset_mode="none",
            )
            session_queue = SessionQueue(max_concurrent=10)
            skill_registry = SkillRegistry()
            health_store = TokenHealthStore(str(workspace_path))
            
            token_policy = DynamicTokenPolicy(
                token_pool,
                strategy="health",
                primary_token_id="test-token-1",
            )
            agent_pool = AgentInstancePool(max_concurrent_per_instance=4)
            token_interceptor = TokenHealthInterceptor(token_pool, health_store)
            
            # 加载agent配置
            agent_loader = AgentLoader(str(workspace_path))
            agent_config = agent_loader.load_agent("main")
            
            # 创建Agent
            def _create_model(token: TokenEntry):
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.providers.openai import OpenAIProvider
                provider = OpenAIProvider(api_key=token.api_key, base_url=token.base_url)
                return OpenAIChatModel(token.model, provider=provider)
            
            model = _create_model(token_entry)
            agent = Agent(
                model,
                deps_type=SkillDeps,
                system_prompt=agent_config.system_prompt or "You are a helpful assistant.",
            )
            
            # 创建AgentRunner
            prompt_builder = PromptBuilder(PromptBuilderConfig())
            
            def agent_factory(agent_id: str, token: TokenEntry):
                return Agent(
                    _create_model(token),
                    deps_type=SkillDeps,
                    system_prompt=agent_config.system_prompt or "You are a helpful assistant.",
                )
            
            agent_runner = AgentRunner(
                agent=agent,
                session_manager=session_manager,
                prompt_builder=prompt_builder,
                session_queue=session_queue,
                agent_id="main",
                token_policy=token_policy,
                agent_pool=agent_pool,
                token_interceptor=token_interceptor,
                agent_factory=agent_factory,
            )
            
            # 执行LLM调用
            session_key = "test-simple-llm-session"
            user_message = "Say 'Hello World' and nothing else."
            
            events = []
            full_response = ""
            error_occurred = False
            error_message = ""
            
            deps = SkillDeps(
                peer_id="default",
                session_key=session_key,
                channel="api",
            )
            
            async for event in agent_runner.run(
                session_key=session_key,
                user_message=user_message,
                deps=deps,
                timeout_seconds=60,
            ):
                events.append(event)
                
                if event.type == "assistant":
                    full_response += event.content or ""
                elif event.type == "error":
                    error_occurred = True
                    error_message = event.error or "Unknown error"
            
            # 验证结果
            assert not error_occurred, f"LLM call failed with error: {error_message}"
            assert len(events) > 0, "No events received"
            assert len(full_response.strip()) > 0, "Empty response from LLM"
            
            # 验证事件流包含必要的阶段
            event_types = [e.type for e in events]
            assert "lifecycle" in event_types, "Missing lifecycle events"
            assert "assistant" in event_types, "Missing assistant response"
            
            print(f"\n=== Simple LLM Call Test ===")
            print(f"Events received: {len(events)}")
            print(f"Response: {full_response[:200]}{'...' if len(full_response) > 200 else ''}")
            print(f"Test PASSED!")

    @pytest.mark.llm
    def test_api_endpoints_respond(self):
        """
        验证API端点正常响应
        
        验证：
        1. 健康检查端点正常
        2. Session创建端点正常
        3. Agent run端点接受请求
        
        注意：BackgroundTasks在TestClient中不会自动执行，
        所以这里只验证API端点响应，不验证实际LLM调用。
        实际LLM调用由 test_simple_agent_call_to_llm 测试覆盖。
        """
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from fastapi.testclient import TestClient
        
        # 加载.env文件
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
        
        # 检查环境变量
        required_vars = ["TOKEN_1_API_KEY", "TOKEN_1_BASE_URL"]
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            pytest.skip(f"Missing environment variables: {missing}")
        
        # 设置环境变量供main.py使用
        os.environ["DEEPSEEK_API_KEY"] = os.environ.get("TOKEN_1_API_KEY", "")
        os.environ["DEEPSEEK_BASE_URL"] = os.environ.get("TOKEN_1_BASE_URL", "")
        
        # 导入app（这会触发lifespan）
        import importlib
        import app.atlasclaw.main as main_module
        importlib.reload(main_module)
        
        app = main_module.app
        
        with TestClient(app) as client:
            # 1. 健康检查
            health_resp = client.get("/api/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "healthy"
            
            # 2. 创建session
            session_resp = client.post("/api/sessions", json={"chat_type": "dm"})
            assert session_resp.status_code == 200
            session_key = session_resp.json()["session_key"]
            assert session_key  # 非空
            
            # 3. Agent run端点接受请求
            run_resp = client.post(
                "/api/agent/run",
                json={
                    "session_key": session_key,
                    "message": "Test message",
                    "timeout_seconds": 60,
                }
            )
            assert run_resp.status_code == 200
            run_id = run_resp.json()["run_id"]
            assert run_id  # 非空
            
            print(f"\n=== API Endpoints Test ===")
            print(f"Health: OK")
            print(f"Session: {session_key}")
            print(f"Run ID: {run_id}")
            print(f"Test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "llm"])
