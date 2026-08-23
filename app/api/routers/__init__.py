"""HTTP routers. One module per resource; `/v1` prefixes live on the routers."""

from app.api.routers.chat import router as chat_router
from app.api.routers.health import router as health_router

__all__ = ["chat_router", "health_router"]
