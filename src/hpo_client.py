"""Minimal HTTP client for the HPO broker.

This is the entire contract a worker needs to participate in a study:
    suggest -> report_epoch (per epoch) -> complete

It has no ML / framework dependencies (no torch, cv2, DeepCrack, or UNet). Bring your own
training loop and call these three methods. The root `colab_worker.py` is the full
bridge-crack reference implementation; cloners should use this client plus
`templates/worker_minimal.py` instead of forking that file.

Environment:
    HPO_BROKER_URL   Base URL of the broker (e.g. an ngrok tunnel). Required for HTTP mode.
    HPO_STUDY_NAME   Optional default study name.

Example:
    from hpo_client import TrialSession

    session = TrialSession()              # reads HPO_BROKER_URL / HPO_STUDY_NAME
    trial = session.suggest()             # {trial_id, trial_number, params}
    for epoch in range(num_epochs):
        dice, bce = train_one_epoch(trial["params"])
        if session.report_epoch(epoch, dice, bce):   # True => broker says prune
            session.complete(epoch, dice, bce, state="PRUNED")
            break
    else:
        session.complete(epoch, dice, bce, weights_path="model.pt", history=session.history)
"""
import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_TIMEOUT = 120




def normalize_broker_url(base_url: str, path: str) -> str:
    """Join a broker base URL and an API path, tolerating trailing /api... in the base."""
    base = (base_url or "").rstrip("/")
    for suffix in ("/api/suggest_trial", "/api/suggest_trials", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if not path.startswith("/"):
        path = "/" + path
    return base + path


class TrialSession:
    """One worker's view of a single trial lifecycle against the broker."""

    def __init__(
        self,
        broker_url: Optional[str] = None,
        study_name: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.broker_url = broker_url or os.getenv("HPO_BROKER_URL")
        if not self.broker_url:
            raise ValueError(
                "broker_url is required (pass it explicitly or set HPO_BROKER_URL)."
            )
        self.study_name = study_name or os.getenv("HPO_STUDY_NAME")
        self.timeout = timeout

        if not self.study_name:
            raise ValueError(
                "study_name is required (pass it explicitly or set HPO_STUDY_NAME)."
            )

        self.trial_id: Optional[int] = None
        self.trial_number: Optional[int] = None
        self.params: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

        import uuid
        self.worker_id = str(uuid.uuid4())
        self._heartbeat_thread = None
        self._heartbeat_active = False
        self.detected_env = self.detect_environment()

    def detect_environment(self) -> Dict[str, Any]:
        """Auto-detect git commit, python version, hostname.
        Call once at session start, pass results to complete()."""
        import sys
        import socket
        import platform
        import subprocess

        env = {
            "python_version": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        }

        # Git commit
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                env["git_commit"] = result.stdout.strip()
        except Exception:
            pass

        # CUDA version (if torch available)
        try:
            import torch
            if torch.cuda.is_available():
                env["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
        except Exception:
            pass

        # pip freeze
        packages = []
        try:
            import importlib.metadata
            packages = [f"{dist.name}=={dist.version}" for dist in importlib.metadata.distributions() if dist.name]
        except Exception:
            try:
                # Fallback to subprocess pip freeze if importlib.metadata fails
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    packages = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            except Exception:
                pass

        if packages:
            packages = sorted(packages)
            if os.getenv("HPO_CAPTURE_FULL_ENV") != "1":
                whitelist = {
                    "torch", "torchvision", "transformers", "accelerate", "numpy", "pandas",
                    "scipy", "scikit-learn", "optuna", "pathfinder", "pydantic", "fastapi",
                    "uvicorn", "sqlalchemy"
                }
                normalized_whitelist = {name.lower().replace("_", "-") for name in whitelist}
                
                filtered_packages = []
                for p in packages:
                    parts = p.split("==")
                    if parts:
                        pkg_name = parts[0].strip()
                        if pkg_name.lower().replace("_", "-") in normalized_whitelist:
                            filtered_packages.append(p)
                packages = filtered_packages
                
            env["pip_freeze"] = "\n".join(packages)

        return env

    # --- low-level HTTP ---
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.broker_url and ("ngrok-free.app" in self.broker_url or "ngrok.io" in self.broker_url):
            headers["ngrok-skip-browser-warning"] = "1"
        secret_token = os.getenv("HPO_SECRET_TOKEN")
        if secret_token:
            headers["X-HPO-Token"] = secret_token
        return headers

    def _is_retryable(self, e: Exception) -> bool:
        if hasattr(e, "response") and e.response is not None:
            status = e.response.status_code
            return status in (408, 429) or status >= 500
        # Network-level exceptions (timeouts, connection refused, etc.) are always retryable
        if isinstance(e, requests.RequestException):
            return True
        return False

    def _post(self, path: str, payload: Dict[str, Any], max_attempts: int = 5) -> Dict[str, Any]:
        import time
        url = normalize_broker_url(self.broker_url, path)
        attempt = 0
        backoff = 1.0
        while True:
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if isinstance(e, requests.exceptions.ConnectionError) and "Connection refused" in str(e):
                    raise RuntimeError(f"Cannot reach HPO broker at {self.broker_url}. Is the server running?") from e
                
                attempt += 1
                if attempt >= max_attempts or not self._is_retryable(e):
                    raise RuntimeError(f"Cannot reach HPO broker at {self.broker_url}. Is the server running? Details: {e}") if isinstance(e, requests.exceptions.ConnectionError) else e
                print(f"Request POST {path} failed: {e}. Retrying in {backoff}s (attempt {attempt}/{max_attempts})...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)

    def _get(self, path: str, timeout: Optional[int] = None, max_attempts: int = 5) -> Dict[str, Any]:
        import time
        url = normalize_broker_url(self.broker_url, path)
        attempt = 0
        backoff = 1.0
        while True:
            try:
                resp = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout or self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if isinstance(e, requests.exceptions.ConnectionError) and "Connection refused" in str(e):
                    raise RuntimeError(f"Cannot reach HPO broker at {self.broker_url}. Is the server running?") from e
                
                attempt += 1
                if attempt >= max_attempts or not self._is_retryable(e):
                    raise RuntimeError(f"Cannot reach HPO broker at {self.broker_url}. Is the server running? Details: {e}") if isinstance(e, requests.exceptions.ConnectionError) else e
                print(f"Request GET {path} failed: {e}. Retrying in {backoff}s (attempt {attempt}/{max_attempts})...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)

    def _run_heartbeat(self):
        import time
        while self._heartbeat_active:
            for _ in range(15):
                if not self._heartbeat_active:
                    return
                time.sleep(1.0)
            if not self._heartbeat_active:
                return
            try:
                payload = {
                    "study_name": self.study_name,
                    "worker_id": self.worker_id,
                    "trial_id": self.trial_id
                }
                requests.post(
                    normalize_broker_url(self.broker_url, "/api/heartbeat"),
                    json=payload,
                    headers=self._get_headers(),
                    timeout=10
                )
            except Exception:
                pass

    # --- broker contract ---
    def health(self) -> Dict[str, Any]:
        """GET /health — quick reachability + endpoint listing check."""
        return self._get("/health", timeout=30)

    def suggest(
        self,
        agent_model: Optional[str] = None,
        prompt_strategy: Optional[str] = None,
        reasoning: Optional[str] = None,
        estimated_score_improvement: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /api/suggest_trial — get the next trial's parameters.

        Stores trial_id / trial_number / params on the session and returns the raw response.
        """
        payload: Dict[str, Any] = {
            "study_name": self.study_name,
            "worker_id": self.worker_id,
        }
        if agent_model is not None:
            payload["agent_model"] = agent_model
        if prompt_strategy is not None:
            payload["prompt_strategy"] = prompt_strategy
        if reasoning is not None:
            payload["reasoning"] = reasoning

        if estimated_score_improvement is not None:
            payload["estimated_score_improvement"] = estimated_score_improvement

        data = self._post("/api/suggest_trial", payload)
        self.trial_id = data.get("trial_id")
        self.trial_number = data.get("trial_number")
        self.params = data.get("params", {})
        self.history = []

        # Start background heartbeat
        if self.trial_id is not None:
            self._heartbeat_active = True
            import threading
            self._heartbeat_thread = threading.Thread(target=self._run_heartbeat, daemon=True)
            self._heartbeat_thread.start()

        return data

    def report_epoch(
        self,
        epoch: int,
        score: float,
        loss: float,
        score_eval_fixed: Optional[float] = None,
        loss_eval_fixed: Optional[float] = None,
        gpu_memory: Optional[float] = None,
        speed_ips: Optional[float] = None,
    ) -> bool:
        """POST /api/report_epoch — log one epoch. Returns True if the broker says prune."""
        if self.trial_id is None:
            raise RuntimeError("Call suggest() before report_epoch().")

        payload: Dict[str, Any] = {
            "study_name": self.study_name,
            "trial_id": self.trial_id,
            "worker_id": self.worker_id,
            "epoch": epoch,
            "score": score,
            "loss": loss,
        }
        if gpu_memory is not None:
            payload["gpu_memory"] = gpu_memory
        if speed_ips is not None:
            payload["speed_ips"] = speed_ips
        if score_eval_fixed is not None:
            payload["score_eval_fixed"] = score_eval_fixed
        if loss_eval_fixed is not None:
            payload["loss_eval_fixed"] = loss_eval_fixed

        entry = {
            "epoch": epoch,
            "score": score,
            "loss": loss,
            # Backwards compatibility keys for UI / charts
            "dice": score,
            "bce": loss,
        }
        if score_eval_fixed is not None:
            entry["score_eval_fixed"] = score_eval_fixed
            entry["dice_eval_fixed"] = score_eval_fixed
            entry["loss_eval_fixed"] = loss_eval_fixed
            entry["bce_eval_fixed"] = loss_eval_fixed
        self.history.append(entry)

        data = self._post("/api/report_epoch", payload)
        return bool(data.get("should_prune", False))

    def complete(
        self,
        epoch: int,
        score: float,
        loss: float,
        weights_path: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
        state: str = "COMPLETE",
        score_eval_fixed: Optional[float] = None,
        loss_eval_fixed: Optional[float] = None,
        gpu_model: Optional[str] = None,
        max_vram_gb: Optional[float] = None,
        oom_triggered: Optional[bool] = None,
        dataset_version: Optional[str] = None,
        git_commit: Optional[str] = None,
        python_version: Optional[str] = None,
        cuda_version: Optional[str] = None,
        pip_freeze: Optional[str] = None,
        hostname: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /api/complete_trial — finalize the trial (COMPLETE, PRUNED, or FAIL)."""
        # Stop background heartbeat
        self._heartbeat_active = False
        if self._heartbeat_thread:
            try:
                self._heartbeat_thread.join(timeout=1.0)
            except Exception:
                pass
        if self.trial_id is None:
            raise RuntimeError("Call suggest() before complete().")

        payload: Dict[str, Any] = {
            "study_name": self.study_name,
            "trial_id": self.trial_id,
            "worker_id": self.worker_id,
            "epoch": epoch,
            "score": score,
            "loss": loss,
            "weights_path": weights_path,
            "history": history if history is not None else self.history,
            "state": state,
        }
        if score_eval_fixed is not None:
            payload["score_eval_fixed"] = score_eval_fixed
        if loss_eval_fixed is not None:
            payload["loss_eval_fixed"] = loss_eval_fixed
        if gpu_model is not None:
            payload["gpu_model"] = gpu_model
        if max_vram_gb is not None:
            payload["max_vram_gb"] = max_vram_gb
        if oom_triggered is not None:
            payload["oom_triggered"] = oom_triggered

        # Auto-detect defaults
        env = dict(self.detected_env) if hasattr(self, "detected_env") else {}
        # Override with explicit values if passed
        if git_commit is not None: env["git_commit"] = git_commit
        if python_version is not None: env["python_version"] = python_version
        if cuda_version is not None: env["cuda_version"] = cuda_version
        if pip_freeze is not None: env["pip_freeze"] = pip_freeze
        if dataset_version is not None: env["dataset_version"] = dataset_version
        if hostname is not None: env["hostname"] = hostname
        if platform is not None: env["platform"] = platform
        
        # Merge into payload
        for k, v in env.items():
            if v is not None:
                payload[k] = v

        data = self._post("/api/complete_trial", payload)

        if os.getenv("HPO_SPARKLINES") == "1":
            try:
                from src.sparklines import print_sparkline

                completed_scores = data.get("completed_scores", [])
                best_score = data.get("best_score", 0.0)
                trial_num = data.get("trial_number", self.trial_number or self.trial_id)
                if completed_scores:
                    print_sparkline(self.study_name, completed_scores, best_score, trial_num)
            except Exception:
                pass

        return data

