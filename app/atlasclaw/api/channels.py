# -*- coding: utf-8 -*-
"""Channel management API routes."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.atlasclaw.channels.manager import ChannelManager
from app.atlasclaw.channels.models import ChannelConnection
from app.atlasclaw.channels.registry import ChannelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/channels", tags=["channels"])

# Global channel manager instance (will be set during app startup)
_channel_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Get channel manager instance."""
    if _channel_manager is None:
        raise HTTPException(status_code=500, detail="Channel manager not initialized")
    return _channel_manager


def set_channel_manager(manager: ChannelManager) -> None:
    """Set channel manager instance."""
    global _channel_manager
    _channel_manager = manager


def get_current_user_id(request: Request) -> str:
    """Get current user ID from request.
    
    For now, returns a default user. In production, this would
    extract user info from authentication.
    """
    # TODO: Implement proper user authentication
    return request.headers.get("X-User-Id", "default")


# Request/Response Models

class ConnectionCreateRequest(BaseModel):
    """Request model for creating a connection."""
    name: str
    config: Dict[str, Any] = {}
    enabled: bool = True
    is_default: bool = False


class ConnectionUpdateRequest(BaseModel):
    """Request model for updating a connection."""
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    is_default: Optional[bool] = None


class ConnectionResponse(BaseModel):
    """Response model for a connection."""
    id: str
    name: str
    channel_type: str
    config: Dict[str, Any]
    enabled: bool
    is_default: bool


class ChannelTypeResponse(BaseModel):
    """Response model for a channel type."""
    type: str
    name: str
    icon: Optional[str] = None
    mode: str
    connection_count: int = 0


class ValidationResponse(BaseModel):
    """Response model for config validation."""
    valid: bool
    errors: List[str] = []


# Routes

@router.get("")
async def list_channel_types(
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> List[ChannelTypeResponse]:
    """List all available channel types with connection counts.
    
    Returns:
        List of channel types with their info
    """
    user_id = get_current_user_id(request)
    channels = ChannelRegistry.list_channels()
    
    result = []
    for channel in channels:
        # Count connections for this channel type
        connections = manager.store.get_connections(user_id, channel["type"])
        
        result.append(ChannelTypeResponse(
            type=channel["type"],
            name=channel.get("name", channel["type"]),
            icon=channel.get("icon"),
            mode=channel.get("mode", "bidirectional"),
            connection_count=len(connections)
        ))
    
    return result


@router.get("/{channel_type}/schema")
async def get_channel_schema(channel_type: str) -> Dict[str, Any]:
    """Get configuration schema for a channel type.
    
    Args:
        channel_type: Channel type identifier
        
    Returns:
        JSON Schema for channel configuration
    """
    handler_class = ChannelRegistry.get(channel_type)
    if not handler_class:
        raise HTTPException(status_code=404, detail=f"Channel type not found: {channel_type}")
    
    # Create temporary instance to get schema
    try:
        handler = handler_class({})
        return handler.describe_schema()
    except Exception as e:
        logger.error(f"Failed to get schema for {channel_type}: {e}")
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


@router.get("/{channel_type}/connections")
async def list_connections(
    channel_type: str,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> Dict[str, Any]:
    """List all connections for a channel type.
    
    Args:
        channel_type: Channel type identifier
        
    Returns:
        List of connections
    """
    user_id = get_current_user_id(request)
    
    handler_class = ChannelRegistry.get(channel_type)
    if not handler_class:
        raise HTTPException(status_code=404, detail=f"Channel type not found: {channel_type}")
    
    connections = manager.store.get_connections(user_id, channel_type)
    
    return {
        "channel_type": channel_type,
        "connections": [
            {
                "id": conn.id,
                "name": conn.name,
                "channel_type": conn.channel_type,
                "config": conn.config,
                "enabled": conn.enabled,
                "is_default": conn.is_default,
            }
            for conn in connections
        ]
    }


@router.post("/{channel_type}/connections")
async def create_connection(
    channel_type: str,
    data: ConnectionCreateRequest,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> ConnectionResponse:
    """Create a new channel connection.
    
    Args:
        channel_type: Channel type identifier
        data: Connection data
        
    Returns:
        Created connection
    """
    user_id = get_current_user_id(request)
    
    handler_class = ChannelRegistry.get(channel_type)
    if not handler_class:
        raise HTTPException(status_code=404, detail=f"Channel type not found: {channel_type}")
    
    # Generate unique ID
    connection_id = f"{channel_type}-{uuid.uuid4().hex[:8]}"
    
    connection = ChannelConnection(
        id=connection_id,
        name=data.name,
        channel_type=channel_type,
        config=data.config,
        enabled=data.enabled,
        is_default=data.is_default,
    )
    
    if not manager.store.save_connection(user_id, channel_type, connection):
        raise HTTPException(status_code=500, detail="Failed to save connection")
    
    return ConnectionResponse(
        id=connection.id,
        name=connection.name,
        channel_type=connection.channel_type,
        config=connection.config,
        enabled=connection.enabled,
        is_default=connection.is_default,
    )


@router.patch("/{channel_type}/connections/{connection_id}")
async def update_connection(
    channel_type: str,
    connection_id: str,
    data: ConnectionUpdateRequest,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> ConnectionResponse:
    """Update an existing channel connection.
    
    Args:
        channel_type: Channel type identifier
        connection_id: Connection identifier
        data: Update data
        
    Returns:
        Updated connection
    """
    user_id = get_current_user_id(request)
    
    connection = manager.store.get_connection(user_id, channel_type, connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail=f"Connection not found: {connection_id}")
    
    # Update fields
    if data.name is not None:
        connection.name = data.name
    if data.config is not None:
        connection.config = data.config
    if data.enabled is not None:
        connection.enabled = data.enabled
    if data.is_default is not None:
        connection.is_default = data.is_default
    
    if not manager.store.save_connection(user_id, channel_type, connection):
        raise HTTPException(status_code=500, detail="Failed to update connection")
    
    return ConnectionResponse(
        id=connection.id,
        name=connection.name,
        channel_type=connection.channel_type,
        config=connection.config,
        enabled=connection.enabled,
        is_default=connection.is_default,
    )


@router.delete("/{channel_type}/connections/{connection_id}")
async def delete_connection(
    channel_type: str,
    connection_id: str,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> JSONResponse:
    """Delete a channel connection.
    
    Args:
        channel_type: Channel type identifier
        connection_id: Connection identifier
        
    Returns:
        Success response
    """
    user_id = get_current_user_id(request)
    
    # Stop connection if active
    await manager.stop_connection(user_id, channel_type, connection_id)
    
    if not manager.store.delete_connection(user_id, channel_type, connection_id):
        raise HTTPException(status_code=404, detail=f"Connection not found: {connection_id}")
    
    return JSONResponse(content={"status": "ok", "message": "Connection deleted"})


@router.post("/{channel_type}/connections/{connection_id}/verify")
async def verify_connection(
    channel_type: str,
    connection_id: str,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> ValidationResponse:
    """Verify a connection's configuration.
    
    Args:
        channel_type: Channel type identifier
        connection_id: Connection identifier
        
    Returns:
        Validation result
    """
    user_id = get_current_user_id(request)
    
    connection = manager.store.get_connection(user_id, channel_type, connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail=f"Connection not found: {connection_id}")
    
    handler_class = ChannelRegistry.get(channel_type)
    if not handler_class:
        raise HTTPException(status_code=404, detail=f"Channel type not found: {channel_type}")
    
    # Create handler instance and validate
    try:
        handler = handler_class(connection.config)
        result = await handler.validate_config(connection.config)
        return ValidationResponse(valid=result.valid, errors=result.errors)
    except Exception as e:
        logger.error(f"Validation failed for {connection_id}: {e}")
        return ValidationResponse(valid=False, errors=[str(e)])


@router.post("/{channel_type}/connections/{connection_id}/enable")
async def enable_connection(
    channel_type: str,
    connection_id: str,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> JSONResponse:
    """Enable a channel connection.
    
    Args:
        channel_type: Channel type identifier
        connection_id: Connection identifier
        
    Returns:
        Success response
    """
    user_id = get_current_user_id(request)
    
    if not await manager.enable_connection(user_id, channel_type, connection_id):
        raise HTTPException(status_code=500, detail="Failed to enable connection")
    
    return JSONResponse(content={"status": "ok", "message": "Connection enabled"})


@router.post("/{channel_type}/connections/{connection_id}/disable")
async def disable_connection(
    channel_type: str,
    connection_id: str,
    request: Request,
    manager: ChannelManager = Depends(get_channel_manager)
) -> JSONResponse:
    """Disable a channel connection.
    
    Args:
        channel_type: Channel type identifier
        connection_id: Connection identifier
        
    Returns:
        Success response
    """
    user_id = get_current_user_id(request)
    
    if not await manager.disable_connection(user_id, channel_type, connection_id):
        raise HTTPException(status_code=500, detail="Failed to disable connection")
    
    return JSONResponse(content={"status": "ok", "message": "Connection disabled"})
