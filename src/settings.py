import os
from typing import Optional, List

class Settings:
    DEFAULT_DB_FILENAME = "hpo_studies.db"

    @property
    def database_url(self) -> str:
        return os.getenv("HPO_DATABASE_URL", f"sqlite:///{self.DEFAULT_DB_FILENAME}")

    @property
    def secret_token(self) -> Optional[str]:
        return os.getenv("HPO_SECRET_TOKEN")

    @secret_token.setter
    def secret_token(self, value: Optional[str]):
        if value is None:
            os.environ.pop("HPO_SECRET_TOKEN", None)
        else:
            os.environ["HPO_SECRET_TOKEN"] = value

    @property
    def allowed_origins_raw(self) -> str:
        return os.getenv("HPO_ALLOWED_ORIGINS", "")

    @property
    def backup_on_start(self) -> bool:
        return os.getenv("HPO_BACKUP_ON_START", "0") in ("1", "true", "True")

    @property
    def tunnel_provider(self) -> Optional[str]:
        return os.getenv("HPO_TUNNEL_PROVIDER")

    @property
    def tunnel_enabled(self) -> bool:
        return os.getenv("HPO_TUNNEL_ENABLED", "0") in ("1", "true", "True")

    @property
    def tunnel_url(self) -> Optional[str]:
        return os.getenv("HPO_TUNNEL_URL")

    @property
    def daemon_enabled(self) -> bool:
        return os.getenv("HPO_DAEMON_ENABLED", "0") in ("1", "true", "True")

    @property
    def study_name(self) -> str:
        return os.getenv("HPO_STUDY_NAME", "")

    @property
    def debug(self) -> bool:
        return os.getenv("HPO_DEBUG", "0") in ("1", "true", "True")

    @property
    def sparklines(self) -> bool:
        return os.getenv("HPO_SPARKLINES", "0") in ("1", "true", "True")

    @property
    def capture_full_env(self) -> bool:
        return os.getenv("HPO_CAPTURE_FULL_ENV", "0") in ("1", "true", "True")

    @property
    def broker_url(self) -> Optional[str]:
        return os.getenv("HPO_BROKER_URL")

    @property
    def allowed_origins(self) -> List[str]:
        origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
        raw = self.allowed_origins_raw
        if raw:
            origins.extend([o.strip() for o in raw.split(",") if o.strip()])
        return origins

settings = Settings()
