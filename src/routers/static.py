import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from ..settings import settings

router = APIRouter()

# Resolve absolute paths relative to this router's module file (src/routers/static.py)
# Up two directories gets us to the project root
_base_dir = Path(__file__).resolve().parent.parent.parent

@router.get("/", response_class=HTMLResponse)
def get_gui():
    gui_path = _base_dir / "web" / "index.html"
    if gui_path.exists():
        with open(gui_path, "r") as f:
            html = f.read()
        # Inject only a NON-SECRET hint so the dashboard knows whether to prompt for login.
        auth_required = "true" if settings.secret_token else "false"
        inject_script = f"<script>window.HPO_AUTH_REQUIRED = {auth_required};</script>"
        if "</head>" in html:
            html = html.replace("</head>", f"{inject_script}</head>", 1)
        else:
            html = inject_script + html
        return HTMLResponse(
            content=html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return HTMLResponse(content="<h3>index.html not found</h3>", status_code=404)

@router.get("/styles.css")
def get_styles():
    styles_path = os.path.join(_base_dir, "web", "styles.css")
    if os.path.exists(styles_path):
        return FileResponse(
            styles_path,
            media_type="text/css",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    raise HTTPException(status_code=404, detail="styles.css not found")

@router.get("/hpo_client.py")
def get_hpo_client():
    client_path = os.path.join(_base_dir, "src", "hpo_client.py")
    if os.path.exists(client_path):
        return FileResponse(client_path, media_type="text/x-python", filename="hpo_client.py")
    raise HTTPException(status_code=404, detail="hpo_client.py not found")

@router.get("/worker_minimal.py")
def get_worker_minimal():
    template_path = os.path.join(_base_dir, "templates", "worker_minimal.py")
    if os.path.exists(template_path):
        return FileResponse(template_path, media_type="text/x-python", filename="worker_minimal.py")
    raise HTTPException(status_code=404, detail="worker_minimal.py not found")
