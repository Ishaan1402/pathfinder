import hmac
import os
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.settings import settings
from src.db_manager import init_db
from src.routers import static_router, worker_router, dashboard_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enforce safe, explicit database schema initialization and migrations on startup
    init_db()
    yield

# Initialize main FastAPI application
app = FastAPI(title="Pathfinder HTTP Broker", lifespan=lifespan)

# Setup CORS middleware — permissive in local dev, locked down when auth is configured
_cors_origins = settings.allowed_origins if settings.secret_token else ["*"]
_cors_credentials = bool(settings.secret_token)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication and Authorization security middleware
@app.middleware("http")
async def security_interceptor(request: Request, call_next):
    """Intercept and authorize incoming requests based on token headers/cookies.
    Static files, worker minimal templates, and health endpoints bypass authorization checks.
    """
    path = request.url.path
    bypass_exact = {"/", "/styles.css", "/health", "/api/login"}
    
    if path in bypass_exact or path.startswith("/js/"):
        return await call_next(request)
        
    secret_token = settings.secret_token
    if not secret_token:
        # No security token configured (local development mode)
        return await call_next(request)
        
    # Extract token from header, Authorization Bearer header, or httpOnly session cookie
    token = request.headers.get("X-HPO-Token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header:
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:]
            else:
                token = auth_header
    if not token:
        token = request.cookies.get("hpo_session")
        
    if not token or not hmac.compare_digest(token, secret_token):
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "Unauthorized: provide a valid token via the X-HPO-Token header, or log in at /api/login to set a session cookie."
            }
        )
        
    return await call_next(request)

# No-Cache middleware to prevent proxies (Cloudflare, Nginx, etc.) from caching dashboard assets
@app.middleware("http")
async def no_cache_static_assets(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path == "/styles.css" or path.startswith("/js/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount routers
app.include_router(static_router)
app.include_router(worker_router)
app.include_router(dashboard_router)

# Mount /js directory for static JS files
_js_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "js")
app.mount("/js", StaticFiles(directory=_js_dir), name="js")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected" if settings.database_url else "disconnected",
        "worker_endpoints": {
            "suggest_trial": "POST /api/suggest_trial",
            "report_epoch": "POST /api/report_epoch",
            "complete_trial": "POST /api/complete_trial",
        },
    }

if __name__ == "__main__":
    import argparse
    import sys
    import uvicorn

    from src.tunneling import (
        run_startup_backup,
        resolve_tunnel_provider,
        ensure_secret_token,
        print_security_banner,
        start_daemon_thread,
        setup_cloudflare_tunnel,
        setup_ngrok_tunnel,
    )

    parser = argparse.ArgumentParser(description="Pathfinder HTTP Broker")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default loopback-only)")
    parser.add_argument("--port", type=int, default=8000, help="Binding port")
    parser.add_argument("--daemon", action="store_true", help="Start background health daemon thread")
    parser.add_argument("--tunnel", action="store_true", help="Start background ngrok tunnel")
    parser.add_argument("--tunnel-provider", default=None, choices=["ngrok", "cloudflare", "none"],
                        help="Tunneling provider (ngrok, cloudflare, or none)")
    parser.add_argument("--tunnel-url", default=None,
                        help="Pre-configured static remote tunnel/broker URL (e.g. Cloudflare custom domain)")
    parser.add_argument("--backup-on-start", action="store_true",
                        help="Perform synchronous database backup at startup")
    args = parser.parse_args()

    # Startup backup
    if args.backup_on_start or settings.backup_on_start:
        run_startup_backup()

    # Resolve tunnel config
    tunnel_provider = resolve_tunnel_provider(args.tunnel, args.tunnel_provider, args.tunnel_url)
    tunnel_requested = tunnel_provider != "none"

    # Ensure auth token for non-loopback / tunnelled setups
    secret_token = ensure_secret_token(args.host, tunnel_requested)
    print_security_banner(secret_token, args.host, args.port)

    # Start background daemon
    if args.daemon or settings.daemon_enabled:
        start_daemon_thread()

    # Set up tunnel provider
    static_tunnel_url = args.tunnel_url or settings.tunnel_url
    if tunnel_provider == "cloudflare" or (static_tunnel_url and tunnel_provider != "ngrok"):
        if not static_tunnel_url:
            print("Error: --tunnel-url or HPO_TUNNEL_URL environment variable must be provided for Cloudflare/static tunnel provider.")
            sys.exit(1)
        setup_cloudflare_tunnel(static_tunnel_url, secret_token)
    elif tunnel_provider == "ngrok":
        setup_ngrok_tunnel(args.port, secret_token)

    uvicorn.run("broker:app", host=args.host, port=args.port, factory=False)
