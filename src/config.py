"""Configuration management with API key protection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""), repr=False)
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-4-hydrogen"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_search_results: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "5")))
    max_pages_per_step: int = field(default_factory=lambda: int(os.getenv("MAX_PAGES_PER_STEP", "3")))
    headless_browser: bool = field(
        default_factory=lambda: os.getenv("HEADLESS_BROWSER", "true").lower() in ("true", "1", "yes")
    )
    reports_dir: Path = field(default_factory=lambda: BASE_DIR / "reports")
    logs_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def has_valid_api_key(self) -> bool:
        key = (self.gemini_api_key or "").strip()
        return bool(key) and key != "your_gemini_api_key_here"

    def masked_api_key(self) -> str:
        key = (self.gemini_api_key or "").strip()
        if len(key) <= 8:
            return "****"
        return f"{key[:4]}...{key[-4:]}"


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
