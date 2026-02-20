from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import re
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env (Pydantic v2).

    This class intentionally exposes `settings.vendors` for bootstrap logic.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Core bot
    bot_token: str = Field(..., alias="BOT_TOKEN")
    admin_ids: List[int] = Field(default_factory=list, alias="ADMIN_IDS")

    # Database
    database_url: str = Field(..., alias="DATABASE_URL")

    # Crypto Pay (optional)
    cryptopay_app_id: Optional[str] = Field(None, alias="CRYPTO_PAY_APP_ID")
    cryptopay_token: Optional[str] = Field(None, alias="CRYPTO_PAY_TOKEN")
    webhook_url: Optional[str] = Field(None, alias="WEBHOOK_URL")


    # Vendor slots (up to 4)
    vendor_1_name: Optional[str] = Field(None, alias="VENDOR_1_NAME")
    vendor_1_url: Optional[str] = Field(None, alias="VENDOR_1_URL")
    vendor_1_key: Optional[str] = Field(None, alias="VENDOR_1_KEY")

    vendor_2_name: Optional[str] = Field(None, alias="VENDOR_2_NAME")
    vendor_2_url: Optional[str] = Field(None, alias="VENDOR_2_URL")
    vendor_2_key: Optional[str] = Field(None, alias="VENDOR_2_KEY")

    vendor_3_name: Optional[str] = Field(None, alias="VENDOR_3_NAME")
    vendor_3_url: Optional[str] = Field(None, alias="VENDOR_3_URL")
    vendor_3_key: Optional[str] = Field(None, alias="VENDOR_3_KEY")

    vendor_4_name: Optional[str] = Field(None, alias="VENDOR_4_NAME")
    vendor_4_url: Optional[str] = Field(None, alias="VENDOR_4_URL")
    vendor_4_key: Optional[str] = Field(None, alias="VENDOR_4_KEY")

    # Optional defaults
    default_markup_percent: float = Field(0.0, alias="DEFAULT_MARKUP_PERCENT")

    # IMPORTANT: bootstrap_from_env expects settings.vendors
    # We populate it dynamically from VENDOR_1..4 values.
    vendors: List[Dict[str, str]] = Field(default_factory=list)

    @field_validator("admin_ids", mode="before")
    @classmethod
    def _parse_admin_ids(cls, v: Any) -> List[int]:
        if v is None:
            return []
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, (int, float)):
            return [int(v)]
        s = str(v).strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    return [int(x) for x in data]
            except Exception:
                pass
        parts = [p for p in re.split(r"[\s,]+", s) if p]
        return [int(p) for p in parts]

    def model_post_init(self, __context: Any) -> None:
        # Build `vendors` from vendor_1..4 if not explicitly provided
        built: List[Dict[str, str]] = []
        for i in range(1, 5):
            name = getattr(self, f"vendor_{i}_name")
            url = getattr(self, f"vendor_{i}_url")
            key = getattr(self, f"vendor_{i}_key")
            if name and url and key:
                built.append({"name": str(name), "url": str(url), "key": str(key)})
        # Prefer explicit env-provided vendors if any (rare), otherwise use built list
        if not self.vendors:
            self.vendors = built


# Backwards-compatible singleton + loader used by older bot.py
_settings_singleton: Optional[Settings] = None


def load_settings() -> Settings:
    """Return a cached Settings() instance loaded from .env."""
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton


# Convenience alias many modules expect
settings = load_settings()
