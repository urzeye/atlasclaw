# -*- coding: utf-8 -*-
"""WeCom (企业微信) channel handler."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from app.atlasclaw.channels.handler import ChannelHandler
from app.atlasclaw.channels.models import (
    ChannelMode,
    ChannelValidationResult,
    ConnectionStatus,
    InboundMessage,
    OutboundMessage,
    SendResult,
)

logger = logging.getLogger(__name__)


class WeComHandler(ChannelHandler):
    """WeCom (企业微信) channel handler.
    
    Supports:
    - Group robot webhook (outbound only)
    - Application messaging (bidirectional with callback)
    
    Group Robot Webhook:
    - Simple outbound messaging via webhook URL
    - Supports text, markdown, image, news, file messages
    
    Application Messaging:
    - Full bidirectional messaging
    - Requires Corp ID, Agent ID, and Secret
    - Supports message callback for receiving user messages
    """
    
    channel_type = "wecom"
    channel_name = "WeCom"
    channel_icon = "wecom"  # Brand identifier
    channel_mode = ChannelMode.BIDIRECTIONAL
    supports_long_connection = False
    supports_webhook = True
    
    # WeCom API endpoints
    API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"
    TOKEN_URL = f"{API_BASE}/gettoken"
    SEND_MSG_URL = f"{API_BASE}/message/send"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._access_token: Optional[str] = None
        self._token_expires: int = 0
    
    async def setup(self, connection_config: Dict[str, Any]) -> bool:
        """Initialize WeCom handler with configuration.
        
        Args:
            connection_config: Configuration with webhook_url or corpid/secret
            
        Returns:
            True if setup successful
        """
        try:
            self.config.update(connection_config)
            
            webhook_url = self.config.get("webhook_url")
            corpid = self.config.get("corpid")
            
            if not webhook_url and not corpid:
                logger.error("WeCom requires either webhook_url or corpid")
                return False
            
            return True
        except Exception as e:
            logger.error(f"WeCom setup failed: {e}")
            return False
    
    async def start(self, context: Any) -> bool:
        """Start WeCom handler."""
        try:
            self._status = ConnectionStatus.CONNECTED
            logger.info("WeCom handler started")
            return True
        except Exception as e:
            logger.error(f"WeCom start failed: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    async def connect(self) -> bool:
        """Connect to WeCom (get access token for application mode)."""
        try:
            corpid = self.config.get("corpid")
            corpsecret = self.config.get("corpsecret")
            
            if corpid and corpsecret:
                if not await self._get_access_token():
                    logger.error("Failed to get WeCom access token")
                    return False
            
            self._status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"WeCom connect failed: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from WeCom."""
        self._access_token = None
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    async def stop(self) -> bool:
        """Stop WeCom handler."""
        await self.disconnect()
        return True
    
    async def handle_inbound(self, request: Any) -> Optional[InboundMessage]:
        """Handle incoming WeCom message callback.
        
        Args:
            request: WeCom callback data (XML parsed to dict)
            
        Returns:
            Standardized InboundMessage
        """
        try:
            if isinstance(request, str):
                data = json.loads(request)
            else:
                data = request
            
            msg_type = data.get("MsgType", "text")
            
            # Extract content based on message type
            if msg_type == "text":
                content = data.get("Content", "")
            elif msg_type == "image":
                content = data.get("PicUrl", "")
            elif msg_type == "voice":
                content = data.get("MediaId", "")
            else:
                content = json.dumps(data)
            
            return InboundMessage(
                message_id=data.get("MsgId", ""),
                sender_id=data.get("FromUserName", ""),
                sender_name=data.get("FromUserName", "Anonymous"),
                chat_id=data.get("ToUserName", ""),
                channel_type=self.channel_type,
                content=content,
                content_type=msg_type,
                metadata={
                    "AgentID": data.get("AgentID"),
                    "CreateTime": data.get("CreateTime"),
                },
            )
        except Exception as e:
            logger.error(f"Failed to handle WeCom message: {e}")
            return None
    
    async def send_message(self, outbound: OutboundMessage) -> SendResult:
        """Send message to WeCom.
        
        Supports both webhook and application modes.
        """
        try:
            webhook_url = self.config.get("webhook_url")
            
            if webhook_url:
                return await self._send_webhook_message(outbound)
            else:
                return await self._send_app_message(outbound)
                
        except Exception as e:
            logger.error(f"Failed to send WeCom message: {e}")
            return SendResult(success=False, error=str(e))
    
    async def _send_webhook_message(self, outbound: OutboundMessage) -> SendResult:
        """Send message via group robot webhook."""
        import aiohttp
        
        webhook_url = self.config.get("webhook_url")
        
        # Support different message types
        msg_type = outbound.metadata.get("msgtype", "text") if outbound.metadata else "text"
        
        if msg_type == "markdown":
            payload = {
                "msgtype": "markdown",
                "markdown": {"content": outbound.content}
            }
        else:
            payload = {
                "msgtype": "text",
                "text": {"content": outbound.content}
            }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("errcode") == 0:
                        return SendResult(success=True)
                    else:
                        return SendResult(
                            success=False,
                            error=f"WeCom error: {data.get('errmsg')}"
                        )
                else:
                    return SendResult(success=False, error=f"HTTP {response.status}")
    
    async def _send_app_message(self, outbound: OutboundMessage) -> SendResult:
        """Send message via application API."""
        import aiohttp
        
        if not self._access_token:
            if not await self._get_access_token():
                return SendResult(success=False, error="Failed to get access token")
        
        url = f"{self.SEND_MSG_URL}?access_token={self._access_token}"
        
        payload = {
            "touser": outbound.chat_id or "@all",
            "msgtype": "text",
            "agentid": self.config.get("agentid"),
            "text": {"content": outbound.content},
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("errcode") == 0:
                        return SendResult(success=True, message_id=data.get("msgid"))
                    else:
                        return SendResult(
                            success=False,
                            error=f"WeCom error: {data.get('errmsg')}"
                        )
                else:
                    return SendResult(success=False, error=f"HTTP {response.status}")
    
    async def validate_config(self, config: Dict[str, Any]) -> ChannelValidationResult:
        """Validate WeCom configuration."""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return ChannelValidationResult(valid=False, errors=errors)
        
        webhook_url = config.get("webhook_url")
        corpid = config.get("corpid")
        
        if not webhook_url and not corpid:
            errors.append("Either webhook_url or corpid is required")
        
        if corpid:
            if not config.get("corpsecret"):
                errors.append("corpsecret is required when using corpid")
            if not config.get("agentid"):
                errors.append("agentid is required when using corpid")
        
        return ChannelValidationResult(valid=len(errors) == 0, errors=errors)
    
    def describe_schema(self) -> Dict[str, Any]:
        """Return WeCom configuration schema."""
        return {
            "type": "object",
            "title": "WeCom",
            "description": "WeCom bot configuration (Webhook URL or Corp ID required)",
            "oneOf_hint": "webhook_url or corpid",
            "properties": {
                "webhook_url": {
                    "type": "string",
                    "title": "Webhook URL",
                    "description": "Group bot Webhook URL (or use Corp ID)",
                    "placeholder": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
                },
                "corpid": {
                    "type": "string",
                    "title": "Corp ID",
                    "description": "WeCom Corp ID (or use Webhook)",
                    "placeholder": "ww...",
                },
                "corpsecret": {
                    "type": "string",
                    "title": "Corp Secret",
                    "description": "Application secret (required with Corp ID)",
                    "placeholder": "App secret",
                },
                "agentid": {
                    "type": "integer",
                    "title": "Agent ID",
                    "description": "Application Agent ID (required with Corp ID)",
                    "placeholder": "1000001",
                },
            },
        }
    
    async def _get_access_token(self) -> bool:
        """Get WeCom access token."""
        try:
            import aiohttp
            
            if self._access_token and time.time() < self._token_expires:
                return True
            
            params = {
                "corpid": self.config.get("corpid"),
                "corpsecret": self.config.get("corpsecret"),
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.TOKEN_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("errcode") == 0:
                            self._access_token = data.get("access_token")
                            self._token_expires = time.time() + data.get("expires_in", 7200) - 300
                            return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to get WeCom access token: {e}")
            return False
