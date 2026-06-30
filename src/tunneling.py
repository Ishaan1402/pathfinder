import os
import datetime
import sqlite3
import glob
import subprocess
import threading
import time
from typing import Optional

import requests

from .settings import settings
from .db_manager import get_db_session
from .schema import SystemConfiguration


def run_startup_backup() -> None:
    """Create a backup of the SQLite database on broker startup, pruning old backups."""
    try:
        db_url = settings.database_url
        if not db_url.startswith("sqlite"):
            print("Skipping automatic startup backup: DATABASE_URL is not SQLite.")
            return
        db_path = db_url.replace("sqlite:///", "") if db_url.startswith("sqlite:///") else "hpo_studies.db"
        if not os.path.exists(db_path):
            return
        os.makedirs("backups", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = f"backups/hpo_backup_{timestamp}.db"
        src_conn = sqlite3.connect(db_path)
        dst_conn = sqlite3.connect(dest_path)
        with dst_conn:
            src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        print(f"\u2713 Automatic startup backup created: {dest_path}")

        backup_files = sorted(glob.glob("backups/hpo_backup_*.db"))
        if len(backup_files) > 10:
            for f in backup_files[:-10]:
                try:
                    os.remove(f)
                    print(f"Deleted old backup: {f}")
                except Exception as remove_err:
                    print(f"Failed to delete old backup {f}: {remove_err}")
    except Exception as e:
        print(f"Failed to run automatic startup backup: {e}")


def _persist_tunnel_url(url: str, secret_token: Optional[str], label: str = "") -> None:
    """Persist the tunnel URL in SQLite for dashboard discovery."""
    try:
        with get_db_session() as session:
            session.merge(SystemConfiguration(
                study_name="_global",
                config_key="remote_broker_url",
                config_value=url,
            ))
            session.merge(SystemConfiguration(
                study_name="_global",
                config_key="ngrok_tunnel_url",
                config_value=url,
            ))
            session.commit()
        label_text = f" ({label})" if label else ""
        print(f"\n{'='*50}")
        print(f"\U0001f525 Remote broker URL established{label_text}: {url}")
        if secret_token:
            print(f"   Auto-login Link: {url}/?token={secret_token}")
        print(f"{'='*50}\n")
    except Exception as db_err:
        print(f"Error saving remote broker URL: {db_err}")


def setup_cloudflare_tunnel(url: str, secret_token: Optional[str]) -> None:
    """Persist a pre-configured Cloudflare or static tunnel URL."""
    _persist_tunnel_url(url, secret_token, label="Static/Cloudflare")


def _start_ngrok(port: int, secret_token: Optional[str]) -> Optional[str]:
    """Spawn ngrok, wait for its agent API to report the public URL, return it."""
    try:
        print(f"Spawning ngrok tunnel for port {port}...")
        subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)
        for _ in range(10):
            try:
                res = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=2)
                if res.status_code == 200:
                    tunnels = res.json().get("tunnels", [])
                    for t in tunnels:
                        if t.get("proto") == "https":
                            public_url = t.get("public_url")
                            _persist_tunnel_url(public_url, secret_token, label="ngrok")
                            return public_url
            except Exception:
                time.sleep(1.0)
        print("Warning: Ngrok started but local agent API did not report tunnel URL.")
    except FileNotFoundError:
        print("Warning: 'ngrok' command not found in PATH.")
    except Exception as e:
        print(f"Failed to start ngrok: {e}")
    return None


def setup_ngrok_tunnel(port: int, secret_token: Optional[str]) -> threading.Thread:
    """Start ngrok in a background daemon thread."""
    t = threading.Thread(target=_start_ngrok, args=(port, secret_token), daemon=True)
    t.start()
    return t


def resolve_tunnel_provider(args_tunnel: bool = False,
                            args_tunnel_provider: Optional[str] = None,
                            args_tunnel_url: Optional[str] = None) -> str:
    """Resolve which tunneling provider to use from CLI args, env vars, and settings."""
    provider = args_tunnel_provider
    if provider is None:
        if settings.tunnel_provider:
            provider = settings.tunnel_provider.lower()
        elif args_tunnel or settings.tunnel_enabled:
            provider = "ngrok"
        else:
            provider = "none"

    static_url = args_tunnel_url or settings.tunnel_url
    if static_url and provider == "none":
        provider = "cloudflare"

    return provider or "none"


def ensure_secret_token(args_host: str, tunnel_requested: bool) -> Optional[str]:
    """Auto-generate a secret token if binding beyond loopback or creating a tunnel."""
    secret_token = settings.secret_token
    is_loopback = args_host in ("127.0.0.1", "localhost", "::1")

    if (not is_loopback or tunnel_requested) and not secret_token:
        import secrets
        secret_token = secrets.token_urlsafe(32)
        settings.secret_token = secret_token
        os.environ["HPO_SECRET_TOKEN"] = secret_token
        print("\n" + "!" * 80)
        print("\u26a0\ufe0f  No HPO_SECRET_TOKEN environment variable set.")
        print("   Auto-generating a secure random token for this session.")
        print("!" * 80 + "\n")
    return secret_token


def print_security_banner(secret_token: Optional[str], host: str, port: int) -> None:
    """Print the token and login link banner on startup."""
    if not secret_token:
        return
    is_loopback = host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::")
    host_display = "localhost" if is_loopback else host
    port_suffix = f":{port}" if port != 80 else ""
    local_login_url = f"http://{host_display}{port_suffix}/?token={secret_token}"

    print("\n" + "=" * 80)
    print("\U0001f511 PATHFINDER DASHBOARD SECURITY ACTIVE")
    print(f"   Access Token: {secret_token}")
    print(f"   Auto-login Link: {local_login_url}")
    if not is_loopback:
        print(f"\U0001f512 Secure Private VPN/Tailscale Network Mode enabled. Binding to {host}:{port}")
    print("=" * 80 + "\n")


def start_daemon_thread() -> Optional[threading.Thread]:
    """Start the background health daemon. Caller decides whether to invoke."""
    from .hpo_daemon import run_daemon_loop
    t = threading.Thread(target=run_daemon_loop, kwargs={"interval_seconds": 10}, daemon=True)
    t.start()
    return t
