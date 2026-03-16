# -*- coding: utf-8 -*-
"""Tests for WeCom channel handler."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

from app.atlasclaw.channels.handlers.wecom import WeComHandler
from app.atlasclaw.channels.models import (
    ChannelMode,
    ConnectionStatus,
    InboundMessage,
    OutboundMessage,
    SendResult,
)


class TestWeComHandler:
    """Tests for WeComHandler class."""

    def test_handler_class_attributes(self):
        """Test handler class has correct attributes."""
        assert WeComHandler.channel_type == "wecom"
        assert WeComHandler.channel_name == "WeCom"
        assert WeComHandler.channel_mode == ChannelMode.BIDIRECTIONAL
        assert WeComHandler.supports_long_connection is True
        assert WeComHandler.supports_webhook is True

    def test_handler_init(self):
        """Test handler initialization."""
        handler = WeComHandler()
        assert handler.config == {}
        assert handler._status == ConnectionStatus.DISCONNECTED
        assert handler._ws_client is None

    @pytest.mark.asyncio
    async def test_setup_with_bot_id(self):
        """Test setup with bot_id and secret for WebSocket mode."""
        handler = WeComHandler()
        config = {
            "bot_id": "test_bot_id",
            "secret": "test_secret",
        }
        result = await handler.setup(config)
        assert result is True
        assert handler.config["bot_id"] == "test_bot_id"
        assert handler.config["secret"] == "test_secret"

    @pytest.mark.asyncio
    async def test_setup_with_webhook_url(self):
        """Test setup with webhook_url."""
        handler = WeComHandler()
        config = {
            "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
        }
        result = await handler.setup(config)
        assert result is True
        assert handler.config["webhook_url"].startswith("https://")

    @pytest.mark.asyncio
    async def test_setup_with_corpid(self):
        """Test setup with corpid, corpsecret, and agentid."""
        handler = WeComHandler()
        config = {
            "corpid": "ww123456",
            "corpsecret": "test_secret",
            "agentid": 1000001,
        }
        result = await handler.setup(config)
        assert result is True
        assert handler.config["corpid"] == "ww123456"

    @pytest.mark.asyncio
    async def test_setup_missing_all_config(self):
        """Test setup fails when no valid config provided."""
        handler = WeComHandler()
        config = {}
        result = await handler.setup(config)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_config_valid_bot_id(self):
        """Test config validation with valid bot_id config."""
        handler = WeComHandler()
        config = {
            "bot_id": "test_bot_id",
            "secret": "test_secret",
        }
        result = await handler.validate_config(config)
        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_valid_webhook(self):
        """Test config validation with valid webhook config."""
        handler = WeComHandler()
        config = {
            "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
        }
        result = await handler.validate_config(config)
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_config_valid_corpid(self):
        """Test config validation with valid corpid config."""
        handler = WeComHandler()
        config = {
            "corpid": "ww123456",
            "corpsecret": "test_secret",
            "agentid": 1000001,
        }
        result = await handler.validate_config(config)
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_config_missing_all(self):
        """Test config validation fails with empty config."""
        handler = WeComHandler()
        config = {}
        result = await handler.validate_config(config)
        assert result.valid is False
        assert any("bot_id" in e or "webhook_url" in e or "corpid" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_config_missing_secret(self):
        """Test config validation fails when bot_id provided without secret."""
        handler = WeComHandler()
        config = {
            "bot_id": "test_bot_id",
        }
        result = await handler.validate_config(config)
        assert result.valid is False
        assert any("secret" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_config_missing_corpsecret(self):
        """Test config validation fails when corpid provided without corpsecret."""
        handler = WeComHandler()
        config = {
            "corpid": "ww123456",
        }
        result = await handler.validate_config(config)
        assert result.valid is False
        assert any("corpsecret" in e for e in result.errors)

    def test_describe_schema(self):
        """Test schema description returns valid structure."""
        handler = WeComHandler()
        schema = handler.describe_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "bot_id" in schema["properties"]
        assert "bot_secret" in schema["properties"]
        assert "webhook_url" in schema["properties"]
        assert "corpid" in schema["properties"]
        assert "corpsecret" in schema["properties"]
        assert "agentid" in schema["properties"]

    @pytest.mark.asyncio
    async def test_handle_inbound_text_message(self):
        """Test handling inbound text message (API mode)."""
        handler = WeComHandler()
        request = {
            "MsgType": "text",
            "MsgId": "msg_123",
            "FromUserName": "user_456",
            "ToUserName": "corp_789",
            "Content": "Hello WeCom",
            "CreateTime": 1234567890,
            "AgentID": 1000001,
        }
        
        message = await handler.handle_inbound(request)
        
        assert message is not None
        assert message.message_id == "msg_123"
        assert message.content == "Hello WeCom"
        assert message.sender_id == "user_456"
        assert message.content_type == "text"

    @pytest.mark.asyncio
    async def test_handle_inbound_json_string(self):
        """Test handling inbound message from JSON string."""
        handler = WeComHandler()
        request = json.dumps({
            "MsgType": "text",
            "MsgId": "msg_abc",
            "FromUserName": "user_def",
            "ToUserName": "corp_ghi",
            "Content": "Hello from JSON",
        })
        
        message = await handler.handle_inbound(request)
        
        assert message is not None
        assert message.content == "Hello from JSON"

    @pytest.mark.asyncio
    async def test_handle_inbound_image_message(self):
        """Test handling inbound image message."""
        handler = WeComHandler()
        request = {
            "MsgType": "image",
            "MsgId": "msg_123",
            "FromUserName": "user_456",
            "ToUserName": "corp_789",
            "PicUrl": "https://example.com/image.jpg",
        }
        
        message = await handler.handle_inbound(request)
        
        assert message is not None
        assert message.content_type == "image"
        assert "example.com" in message.content

    @pytest.mark.asyncio
    async def test_start_sets_connecting_status(self):
        """Test start method sets status to CONNECTING."""
        handler = WeComHandler()
        result = await handler.start(None)
        
        assert result is True
        assert handler._status == ConnectionStatus.CONNECTING

    @pytest.mark.asyncio
    async def test_stop_disconnects(self):
        """Test stop method disconnects handler."""
        handler = WeComHandler()
        handler._running = True
        result = await handler.stop()
        
        assert result is True
        assert handler._status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send_message_via_webhook(self):
        """Test sending message via webhook."""
        handler = WeComHandler({
            "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
        })
        
        outbound = OutboundMessage(
            chat_id="",
            content="Test message",
            content_type="text",
        )
        
        with patch("app.atlasclaw.channels.handlers.wecom.aiohttp") as mock_aiohttp:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "errcode": 0,
                "errmsg": "ok"
            })
            
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_post_cm.__aexit__.return_value = None
            
            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm
            
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            
            mock_aiohttp.ClientSession.return_value = mock_session_cm
            
            result = await handler.send_message(outbound)
            
            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_no_method_available(self):
        """Test sending message fails when no method available."""
        handler = WeComHandler()
        
        outbound = OutboundMessage(
            chat_id="user_123",
            content="Test message",
            content_type="text",
        )
        
        result = await handler.send_message(outbound)
        
        assert result.success is False
        assert "No send method available" in result.error


class TestWeComHandlerMessageCallback:
    """Tests for WeCom handler message callback functionality."""

    def test_set_message_callback(self):
        """Test setting message callback."""
        handler = WeComHandler()
        callback = MagicMock()
        
        handler.set_message_callback(callback)
        
        assert handler._message_callback == callback

    @pytest.mark.asyncio
    async def test_handle_message_calls_callback(self):
        """Test that _handle_message calls the callback."""
        handler = WeComHandler()
        callback = MagicMock()
        handler.set_message_callback(callback)
        
        frame = {
            "headers": {"req_id": "test_req_id", "userid": "user_123"},
            "body": {
                "text": {"content": "Hello"},
                "chatid": "chat_456",
                "chattype": "single",
            },
        }
        
        await handler._handle_message(frame, "text")
        
        assert callback.called
        call_args = callback.call_args[0][0]
        assert isinstance(call_args, InboundMessage)
        assert call_args.content == "Hello"
