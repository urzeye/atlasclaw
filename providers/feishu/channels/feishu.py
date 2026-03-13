# -*- coding: utf-8 -*-
"""Feishu (Lark) channel handler."""

from __future__ import annotations

import json
import logging
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


class FeishuHandler(ChannelHandler):
    """Feishu channel handler using WebSocket long connection.
    
    Connects to Feishu Event Center via WebSocket for real-time messaging.
    """
    
    channel_type = "feishu"
    channel_name = "Feishu"
    channel_icon = "feishu"  # Brand identifier
    channel_mode = ChannelMode.BIDIRECTIONAL
    supports_long_connection = True
    supports_webhook = False
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._access_token: Optional[str] = None
        self._websocket: Optional[Any] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
    
    async def setup(self, connection_config: Dict[str, Any]) -> bool:
        """Initialize Feishu handler with configuration.
        
        Args:
            connection_config: Configuration with app_id, app_secret
            
        Returns:
            True if setup successful
        """
        try:
            self.config.update(connection_config)
            
            # Validate required fields
            if not self.config.get("app_id"):
                logger.error("Feishu app_id is required")
                return False
            if not self.config.get("app_secret"):
                logger.error("Feishu app_secret is required")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Feishu setup failed: {e}")
            return False
    
    async def start(self, context: Any) -> bool:
        """Start Feishu handler.
        
        Args:
            context: Application context
            
        Returns:
            True if started successfully
        """
        try:
            self._status = ConnectionStatus.CONNECTED
            logger.info("Feishu handler started")
            return True
        except Exception as e:
            logger.error(f"Feishu start failed: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    async def connect(self) -> bool:
        """Establish WebSocket connection to Feishu Event Center.
        
        Returns:
            True if connected successfully
        """
        try:
            import aiohttp
            
            # Step 1: Get access token
            if not await self._get_access_token():
                logger.error("Failed to get Feishu access token")
                return False
            
            # Step 2: Get WebSocket endpoint
            ws_url = await self._get_websocket_url()
            if not ws_url:
                logger.error("Failed to get Feishu WebSocket URL")
                return False
            
            # Step 3: Connect WebSocket
            # Note: Actual WebSocket connection would be implemented here
            # This is a simplified version for demonstration
            logger.info(f"Connecting to Feishu WebSocket: {ws_url}")
            
            self._status = ConnectionStatus.CONNECTED
            self._reconnect_attempts = 0
            logger.info("Feishu WebSocket connected")
            return True
            
        except Exception as e:
            logger.error(f"Feishu connect failed: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Feishu.
        
        Returns:
            True if disconnected successfully
        """
        try:
            if self._websocket:
                # Close WebSocket connection
                await self._websocket.close()
                self._websocket = None
            
            self._access_token = None
            self._status = ConnectionStatus.DISCONNECTED
            logger.info("Feishu disconnected")
            return True
        except Exception as e:
            logger.error(f"Feishu disconnect failed: {e}")
            return False
    
    async def reconnect(self) -> bool:
        """Reconnect to Feishu after connection loss.
        
        Returns:
            True if reconnected successfully
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"Feishu reconnection attempt {self._reconnect_attempts}")
        
        await self.disconnect()
        return await self.connect()
    
    async def stop(self) -> bool:
        """Stop Feishu handler and cleanup resources.
        
        Returns:
            True if stopped successfully
        """
        try:
            await self.disconnect()
            self._status = ConnectionStatus.DISCONNECTED
            return True
        except Exception as e:
            logger.error(f"Feishu stop failed: {e}")
            return False
    
    async def handle_inbound(self, request: Any) -> Optional[InboundMessage]:
        """Handle incoming Feishu message.
        
        Args:
            request: Feishu event data
            
        Returns:
            Standardized InboundMessage
        """
        try:
            if isinstance(request, str):
                data = json.loads(request)
            else:
                data = request
            
            # Parse Feishu message format
            event_type = data.get("header", {}).get("event_type", "")
            
            if event_type not in ["im.message.receive_v1"]:
                return None
            
            event_data = data.get("event", {})
            message = event_data.get("message", {})
            sender = event_data.get("sender", {})
            
            # Extract message content
            content = message.get("content", "")
            if isinstance(content, str):
                try:
                    content_obj = json.loads(content)
                    text = content_obj.get("text", "")
                except:
                    text = content
            else:
                text = str(content)
            
            return InboundMessage(
                message_id=message.get("message_id", ""),
                sender_id=sender.get("sender_id", {}).get("open_id", ""),
                sender_name=sender.get("nickname", "Anonymous"),
                chat_id=message.get("chat_id", ""),
                channel_type=self.channel_type,
                content=text,
                content_type="text",
                thread_id=message.get("thread_id"),
                metadata={
                    "msg_type": message.get("msg_type"),
                    "create_time": message.get("create_time"),
                },
            )
        except Exception as e:
            logger.error(f"Failed to handle Feishu message: {e}")
            return None
    
    async def send_message(self, outbound: OutboundMessage) -> SendResult:
        """Send message to Feishu.
        
        Args:
            outbound: Outbound message
            
        Returns:
            SendResult with success status
        """
        try:
            import aiohttp
            
            if not self._access_token:
                return SendResult(
                    success=False,
                    error="Not authenticated with Feishu"
                )
            
            url = "https://open.feishu.cn/open-apis/im/v1/messages"
            
            payload = {
                "receive_id": outbound.chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": outbound.content}),
            }
            
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == 0:
                            return SendResult(
                                success=True,
                                message_id=data.get("data", {}).get("message_id")
                            )
                        else:
                            return SendResult(
                                success=False,
                                error=f"Feishu API error: {data.get('msg')}"
                            )
                    else:
                        return SendResult(
                            success=False,
                            error=f"HTTP {response.status}"
                        )
                        
        except Exception as e:
            logger.error(f"Failed to send Feishu message: {e}")
            return SendResult(success=False, error=str(e))
    
    async def validate_config(self, config: Dict[str, Any]) -> ChannelValidationResult:
        """Validate Feishu configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result
        """
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return ChannelValidationResult(valid=False, errors=errors)
        
        if not config.get("app_id"):
            errors.append("app_id is required")
        
        if not config.get("app_secret"):
            errors.append("app_secret is required")
        
        return ChannelValidationResult(valid=len(errors) == 0, errors=errors)
    
    def describe_schema(self) -> Dict[str, Any]:
        """Return Feishu configuration schema.
        
        Returns:
            JSON Schema
        """
        return {
            "type": "object",
            "title": "Feishu",
            "description": "Feishu bot configuration (requires app from Feishu Open Platform)",
            "required": ["app_id", "app_secret"],
            "properties": {
                "app_id": {
                    "type": "string",
                    "title": "App ID",
                    "description": "Feishu application App ID",
                    "placeholder": "cli_xxxxxxxxxx",
                },
                "app_secret": {
                    "type": "string",
                    "title": "App Secret",
                    "description": "Feishu application App Secret",
                    "placeholder": "App secret",
                },
                "encrypt_key": {
                    "type": "string",
                    "title": "Encrypt Key",
                    "description": "Message encryption key (optional)",
                    "placeholder": "For message encryption",
                },
                "verification_token": {
                    "type": "string",
                    "title": "Verification Token",
                    "description": "Webhook verification token (optional)",
                    "placeholder": "For webhook verification",
                },
            },
        }
    
    async def _get_access_token(self) -> bool:
        """Get Feishu access token.
        
        Returns:
            True if successful
        """
        try:
            import aiohttp
            
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            
            payload = {
                "app_id": self.config.get("app_id"),
                "app_secret": self.config.get("app_secret"),
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == 0:
                            self._access_token = data.get("tenant_access_token")
                            return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to get Feishu access token: {e}")
            return False
    
    async def _get_websocket_url(self) -> Optional[str]:
        """Get Feishu WebSocket endpoint URL.
        
        Returns:
            WebSocket URL or None
        """
        # In production, this would call Feishu API to get WebSocket URL
        # For now, return a placeholder
        return "wss://ws.feishu.cn/event_center/"
