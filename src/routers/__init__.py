from .static import router as static_router
from .worker import router as worker_router
from .dashboard import router as dashboard_router

__all__ = ["static_router", "worker_router", "dashboard_router"]
