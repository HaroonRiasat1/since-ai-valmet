"""Settings, read from the environment so that no secret or machine path lives in the code."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    web_dir: Path = field(default_factory=lambda: Path(os.environ.get("WEB_DIR", ROOT / "web")))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))

    @property
    def scene_path(self) -> Path:
        return self.web_dir / "data" / "scene.json"


def get_settings() -> Settings:
    return Settings()
