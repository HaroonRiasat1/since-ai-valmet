"""Settings, read from the environment or .env so that no secret or machine path lives in the code."""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")


@dataclass(frozen=True)
class Settings:
    web_dir: Path = field(default_factory=lambda: Path(os.environ.get("WEB_DIR", ROOT / "web")))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    semantic_dir: Path = field(default_factory=lambda: Path(os.environ.get("SEMANTIC_DIR", ROOT / "data" / "semantic")))

    @property
    def scene_path(self) -> Path:
        return self.web_dir / "data" / "scene.json"


def get_settings() -> Settings:
    return Settings()
