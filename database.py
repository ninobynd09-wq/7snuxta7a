from __future__ import annotations
import uuid

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import html as _html

from dotenv import load_dotenv
import httpx
from sqlalchemy import select, func, desc, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


from config import Settings
from models import Base, AppSetting, Vendor, User, Service, VendorService, Order, Transaction, OrderNotification

logger = logging.getLogger(__name__)


async def commit_with_retry(session: AsyncSession, retries: int = 3, delay: float = 0.5) -> None:
    """Commit the current transaction with a small Postgres-safe retry.

    On transient OperationalError (connection blip, deadlock/serialization surfaced
    as OperationalError by the driver), we rollback, back off briefly, and retry.
    """
    retries = max(1, int(retries))
    base_delay = float(delay)
    for i in range(retries):
        try:
            await session.commit()
            return
        except OperationalError as e:
            # Session must be rolled back before it can be used again.
            try:
                await session.rollback()
            except Exception:
                logger.exception(f"Error processing order: {e}")
            if i == retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** i))


async def run_db_operation(op, retries: int = 3, delay: float = 0.5):
    """Run a DB write operation with a small retry for transient OperationalError.

    This is safe for PostgreSQL and prevents background tasks from dying on a
    transient contention or network blip. The `op` callable must contain its own
    transaction unit (flush/commit).
    """
    retries = max(1, int(retries))
    for i in range(retries):
        try:
            return await op()
        except OperationalError:
            if i == retries - 1:
                raise
            await asyncio.sleep(float(delay))

DEFAULT_MARKUP_KEY = "global_markup_percent"

PLATFORM_LIST = [
    "Instagram",
    "TikTok",
    "YouTube",
    "X(Twitter)",
    "Facebook",
    "Telegram",
    "Spotify",
]


# -----------------------------
# Engine / sessions
# -----------------------------
# NOTE:
#   This project uses a global AsyncEngine and async_sessionmaker so other modules
#   can import them directly:
#     from database import engine, async_session
#
#   The objects are defined at the bottom of this file.


# -----------------------------
# Migrations (lightweight)
# -----------------------------
async def _column_exists(*args, **kwargs) -> bool:
    return False
async def _table_exists(*args, **kwargs) -> bool:
    return False
async def ensure_compat(session: AsyncSession) -> None:
    """PostgreSQL deployment: no legacy PRAGMA-based compatibility migrations.

    Use proper migrations (e.g., Alembic) for schema evolution.
    For fresh installs, Base.metadata.create_all() is sufficient.
    """
    return

async def ensure_order_notifications_table(session: AsyncSession) -> None:
    """Ensure order_notifications table exists (for upgraded installs).

    This function does NOT commit; caller controls transaction boundary.
    """
    try:
        await session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS order_notifications (
                    order_id INTEGER PRIMARY KEY,
                    completed_notified INTEGER DEFAULT 0,
                    created_at TEXT
                )
                """
            )
        )
    except Exception:
        logger.exception("Failed to ensure order_notifications table exists")
        raise

async def init_db() -> None:
    """Create tables (if missing) and apply lightweight compatibility ALTERs."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        await ensure_compat(session)


# -----------------------------
# Global settings (markup)
# -----------------------------
async def bootstrap_from_env(settings: Settings, session: AsyncSession) -> None:
    # Markup
    row = (await session.execute(select(AppSetting).where(AppSetting.key == DEFAULT_MARKUP_KEY))).scalars().first()
    if not row:
        session.add(AppSetting(key=DEFAULT_MARKUP_KEY, value=str(settings.default_markup_percent)))
        await commit_with_retry(session)

    # Seed vendors from .env (if provided)
    for cfg in settings.vendors:
        v = (await session.execute(select(Vendor).where(func.lower(Vendor.name) == cfg['name'].lower()))).scalars().first()
        if not v:
            session.add(Vendor(name=cfg['name'], url=cfg['url'], api_key=cfg['key'], is_active=True))
        else:
            v.url = cfg['url']
            v.api_key = cfg['key']
    await commit_with_retry(session)


async def get_markup_percent(session: AsyncSession) -> float:
    row = (await session.execute(select(AppSetting).where(AppSetting.key == DEFAULT_MARKUP_KEY))).scalars().first()
    try:
        return float(row.value) if row else 10.0
    except Exception:
        return 10.0


async def set_markup_percent(session: AsyncSession, value: float) -> None:
    value = float(value)
    row = (await session.execute(select(AppSetting).where(AppSetting.key == DEFAULT_MARKUP_KEY))).scalars().first()
    if row:
        row.value = str(value)
    else:
        session.add(AppSetting(key=DEFAULT_MARKUP_KEY, value=str(value)))
    await commit_with_retry(session)


# -----------------------------
# Global settings (maintenance mode)
# -----------------------------
MAINTENANCE_KEY = "maintenance_mode"


async def is_maintenance_on(session: AsyncSession) -> bool:
    row = (await session.execute(select(AppSetting).where(AppSetting.key == MAINTENANCE_KEY))).scalars().first()
    if not row:
        return False
    v = str(row.value or "").strip().lower()
    return v in ("on", "true", "1", "yes")


async def toggle_maintenance(session: AsyncSession, is_on: bool) -> None:
    val = "on" if bool(is_on) else "off"
    row = (await session.execute(select(AppSetting).where(AppSetting.key == MAINTENANCE_KEY))).scalars().first()
    if row:
        row.value = val
    else:
        session.add(AppSetting(key=MAINTENANCE_KEY, value=val))
    await commit_with_retry(session)


# -----------------------------
# Vendor adapter (SMM API v2 - form data)
# -----------------------------
@dataclass(frozen=True)
class VendorServiceDTO:
    vendor_id: int
    vendor_name: str
    vendor_service_id: str
    vendor_rate: float
    vendor_min: int
    vendor_max: int
    raw_category: str
    description: str
    name: str
    type: str
    platform: str
    category: str
    sub_type: str


class SMMv1Adapter:
    def __init__(self, vendor: Vendor, timeout: float = 30.0) -> None:
        self.vendor = vendor
        self.timeout = timeout

    def _api_url(self) -> str:
        """Return the vendor API URL.

        Some panels (notably JustAnotherPanel) are strict about the /api/v2 path.
        To reduce misconfiguration regressions, we normalize it here.
        """
        url = (self.vendor.url or "").strip()
        if not url:
            return url

        # Enforce /api/v2 for JustAnotherPanel requests.
        if (self.vendor.name or "").strip() == "JustAnotherPanel" and "/api/v2" not in url:
            u = url.rstrip("/")
            # If someone stored a base domain or /api, normalize to /api/v2.
            if u.endswith("/api"):
                u = u[: -len("/api")]
            url = u + "/api/v2"
        return url

    async def _post(self, payload: Dict[str, Any]) -> Any:
        url = self._api_url()
        try:
            if str(payload.get("action") or "").lower() == "status":
                logger.info("Vendor status check: vendor=%s order=%s url=%s", self.vendor.name, payload.get("order"), url)
        except Exception as e:
            logger.exception(f"Error processing order: {e}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=payload)
            r.raise_for_status()
            return r.json()

    async def fetch_services(self) -> List[VendorServiceDTO]:
        raw = await self._post({"key": self.vendor.api_key, "action": "services"})
        logger.info(f"Vendor {self.vendor.name} Raw Response: {str(raw)[:500]}")
        if not isinstance(raw, list):
            raise RuntimeError(f"{self.vendor.name}: unexpected services response: {raw!r}")

        out: List[VendorServiceDTO] = []
        for item in raw:
            try:
                sid = str(item.get("service"))
                name = str(item.get("name") or "").strip()
                type_ = str(item.get("type") or "").strip()
                rate = float(item.get("rate"))
                vmin = int(item.get("min"))
                vmax = int(item.get("max"))
                raw_cat = str(item.get("category") or "Uncategorized")
                desc = str(item.get("description") or item.get("desc") or "").strip()

                platform = classify_platform(f"{raw_cat} {name}") or "Other"
                category = classify_category(name)
                sub_type = classify_sub_type(platform, category, f"{raw_cat} {name} {type_}")

                if not sid or not name or not type_:
                    continue

                out.append(
                    VendorServiceDTO(
                        vendor_id=int(self.vendor.id),
                        vendor_name=self.vendor.name,
                        vendor_service_id=sid,
                        vendor_rate=rate,
                        vendor_min=vmin,
                        vendor_max=vmax,
                        raw_category=raw_cat,
                        description=desc,
                        name=name,
                        type=type_,
                        platform=platform,
                        category=category,
                        sub_type=sub_type,
                    )
                )
            except Exception as e:
                logger.exception(f"Error processing order: {e}")
        return out

    async def add_order(self, vendor_service_id: str, link: str, quantity: int) -> str:
        data = {"key": self.vendor.api_key, "action": "add", "service": vendor_service_id, "link": link, "quantity": str(quantity)}
        resp = await self._post(data)
        if isinstance(resp, dict) and "order" in resp:
            return str(resp["order"])
        raise RuntimeError(f"{self.vendor.name}: add_order failed: {resp!r}")

    async def get_order_status(self, vendor_order_id: str) -> str:
        """Fetch order status using SMM API v2 (form-data).

        Expected response for most panels is a dict containing a 'status' field.
        Example: {'status': 'Completed', ...}
        """
        payload = {"key": self.vendor.api_key, "action": "status", "order": str(vendor_order_id)}
        resp = await self._post(payload)

        # Log raw response for debugging (truncate to keep logs readable).
        try:
            logger.info(
                "Vendor status raw response: vendor=%s order=%s resp=%s",
                self.vendor.name,
                str(vendor_order_id),
                (str(resp)[:800] + "…") if len(str(resp)) > 800 else str(resp),
            )
        except Exception:
            logger.exception("Failed to log vendor status response")

        if isinstance(resp, dict):
            status = resp.get("status") or resp.get("Status")
            if status is not None:
                # Log extracted status too.
                try:
                    logger.info(
                        "Vendor status extracted: vendor=%s order=%s status=%r",
                        self.vendor.name,
                        str(vendor_order_id),
                        status,
                    )
                except Exception:
                    logger.exception("Failed to log extracted vendor status")
                return str(status)

        # Some panels can return strings or unexpected dicts; bubble up for logs.
        raise RuntimeError(f"{self.vendor.name}: status failed: {resp!r}")

# (get_order_status defined above)
# -----------------------------
# Classification helpers
# -----------------------------
def classify_platform(raw: str) -> Optional[str]:
    s = (raw or "").lower()
    if "instagram" in s:
        return "Instagram"
    if "tiktok" in s or "tik tok" in s:
        return "TikTok"
    if "youtube" in s:
        return "YouTube"
    if "twitter" in s:
        return "X(Twitter)"
    if "facebook" in s or " fb" in s:
        return "Facebook"
    if "telegram" in s:
        return "Telegram"
    if "spotify" in s:
        return "Spotify"
    return None


CATEGORY_KEYWORDS = [
    ("Views", ["view", "watch"]),
    ("Followers", ["follower", "follow"]),
    ("Likes", ["like"]),
    ("Comments", ["comment"]),
    ("Shares", ["share"]),
    ("Saves", ["save"]),
    ("Members", ["member", "join"]),
    ("Subscribers", ["subscriber", "subs"]),
    ("Reactions", ["reaction", "react"]),
]


def classify_category(raw: str) -> str:
    s = (raw or "").lower()
    for label, keys in CATEGORY_KEYWORDS:
        if any(k in s for k in keys):
            return label
    return "Other"

def classify_sub_type(platform: str, category: str, raw: str) -> str:
    """Best-effort tagging of a vendor service into one of the fixed UI sub-types.

    The UI expects these final options:
      Followers: "USA - Drop", "USA - Non-Drop", "Non-USA (Drop Only)"
      Likes: "Cheap Likes"
      Views: "APV", "RAV"

    Vendors label these differently, so we use keyword heuristics. You can still
    override/match by setting Service.sub_type manually in the DB if needed.
    """
    s = (raw or "").lower()

    if category == "Followers":
        usa = any(k in s for k in [" usa", "us ", "united states", "(usa", "usa-"]) or s.startswith("usa")
        nondrop = any(k in s for k in ["non-drop", "non drop", "refill", "guaranteed", "no drop"])
        drop = any(k in s for k in [" drop", "drop ", "no refill", "non refill"])
        if usa and nondrop:
            return "USA - Non-Drop"
        if usa and (drop or not nondrop):
            return "USA - Drop"
        # Non-USA / Global
        globalish = any(k in s for k in ["global", "worldwide", "international", "non usa", "non-usa"]) and (drop or not nondrop)
        if (not usa and (drop or globalish)):
            return "Non-USA (Drop Only)"
        return ""

    if category == "Likes":
        if any(k in s for k in ["cheap", "budget", "economy", "low"]):
            return "Cheap Likes"
        return ""

    if category == "Views":
        if any(k in s for k in ["apv", "bot view", "bot", "artificial", "automated"]):
            return "APV"
        if any(k in s for k in ["rav", "real", "active", "organic"]):
            return "RAV"
        return ""

    return ""



# -----------------------------
# Pricing helpers
# -----------------------------
def apply_markup(rate_per_1000: float, markup_percent: float) -> float:
    return rate_per_1000 * (1.0 + (markup_percent / 100.0))


def calc_charge(rate_per_1000_with_markup: float, quantity: int) -> float:
    return (rate_per_1000_with_markup / 1000.0) * float(quantity)


# -----------------------------
# CRUD helpers
# -----------------------------
async def get_or_create_user(session: AsyncSession, telegram_id: int, username: Optional[str]) -> User:
    user = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalars().first()
    if not user:
        user = User(telegram_id=telegram_id, username=username, balance=0.0)
        session.add(user)
        await commit_with_retry(session)
        await session.refresh(user)
    else:
        if username and user.username != username:
            user.username = username
            await commit_with_retry(session)
    return user


async def find_user_by_telegram_id(session: AsyncSession, telegram_id: int) -> Optional[User]:
    return (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalars().first()


async def add_funds(session: AsyncSession, telegram_id: int, amount: float) -> User:
    for _attempt in range(3):
        try:
            user = await find_user_by_telegram_id(session, telegram_id)
            if not user:
                user = User(telegram_id=telegram_id, username=None, balance=0.0)
                session.add(user)
                await session.flush()
            user.balance = float(user.balance) + float(amount)
            await session.commit()
            await session.refresh(user)
            return user
        except OperationalError as e:
            if _attempt < 2:
                try:
                    await session.rollback()
                except Exception as e:
                    logger.exception(f"Error processing order: {e}")
                await asyncio.sleep(0.5)
                continue
            raise


async def list_platform_categories(session: AsyncSession, platform: str) -> List[str]:
    stmt = select(Service.category).where(Service.platform == platform, Service.is_active == True).distinct().order_by(Service.category.asc())
    return [r[0] for r in (await session.execute(stmt)).all()]


async def list_services(session: AsyncSession, platform: str, category: str, limit: int = 30) -> List[Service]:
    stmt = (
        select(Service)
        .where(Service.platform == platform, Service.category == category, Service.is_active == True)
        .order_by(Service.name.asc(), Service.type.asc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def list_all_vendors(session: AsyncSession) -> List[Vendor]:
    return list((await session.execute(select(Vendor).order_by(Vendor.id.asc()))).scalars().all())


async def add_vendor(session: AsyncSession, name: str, url: str, key: str) -> Vendor:
    v = Vendor(name=name.strip(), url=url.strip(), api_key=key.strip(), is_active=True)
    session.add(v)
    await commit_with_retry(session)
    await session.refresh(v)
    return v


async def toggle_vendor(session: AsyncSession, vendor_id: int) -> Vendor:
    v = (await session.execute(select(Vendor).where(Vendor.id == vendor_id))).scalars().first()
    if not v:
        raise RuntimeError("Vendor not found.")
    v.is_active = not bool(v.is_active)
    await commit_with_retry(session)
    await session.refresh(v)
    return v


async def test_vendor(session: AsyncSession, vendor_id: int) -> Tuple[bool, str]:
    v = (await session.execute(select(Vendor).where(Vendor.id == vendor_id))).scalars().first()
    if not v:
        raise RuntimeError("Vendor not found.")
    adapter = SMMv1Adapter(v)
    try:
        svcs = await adapter.fetch_services()
        v.last_test_at = datetime.now(timezone.utc)
        v.last_test_ok = True
        v.last_error = None
        v.last_services_count = len(svcs)
        await commit_with_retry(session)
        return True, f"OK ({len(svcs)} services)"
    except Exception as e:
        v.last_test_at = datetime.now(timezone.utc)
        v.last_test_ok = False
        v.last_error = str(e)[:500]
        await commit_with_retry(session)
        return False, f"FAILED: {e}"


# -----------------------------
# Sync catalog (active vendors only)
# -----------------------------
def _svc_key(platform: str, name: str, type_: str, sub_type: str) -> Tuple[str, str, str, str]:
    return (platform.strip().lower(), name.strip().lower(), type_.strip().lower(), (sub_type or "").strip().lower())

def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int,)):
            return int(v)
        s = str(v).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _clean_desc(desc: str) -> str:
    """Normalize vendor description to plain text for light parsing."""
    s = (desc or "").strip()
    if not s:
        return ""
    # Remove basic HTML breaks and normalize whitespace
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"[\t ]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_specs_from_description(desc: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort extraction of Speed/Refill/Start Time from vendor descriptions.

    This is intentionally conservative: it only adds metadata and must never
    break sync if patterns are missing.
    """
    s = _clean_desc(desc)
    if not s:
        return None, None, None

    s_l = s.lower()

    def _pick(m: Optional[re.Match]) -> Optional[str]:
        if not m:
            return None
        val = (m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0))
        val = str(val or "").strip()
        if not val:
            return None
        # Trim very long captures
        return val[:60]

    # --- Speed ---
    speed: Optional[str] = None
    m = re.search(r"(?:^|\b)speed\s*[:\-\–\—]\s*([^\n\r\|]+)", s, flags=re.IGNORECASE)
    speed = _pick(m)
    if not speed:
        m2 = re.search(r"\b\d+(?:\.\d+)?\s*[kKmM]?\s*/\s*(?:day|d|hr|hrs|hour|hours)\b", s, flags=re.IGNORECASE)
        speed = _pick(m2)
    if not speed and "instant" in s_l:
        speed = "Instant"

    # --- Refill ---
    refill: Optional[str] = None
    m = re.search(r"(?:^|\b)refill\s*[:\-\–\—]\s*([^\n\r\|]+)", s, flags=re.IGNORECASE)
    refill = _pick(m)
    if not refill:
        if "no refill" in s_l or "non refill" in s_l:
            refill = "No Refill"
        elif "non-drop" in s_l or "nondrop" in s_l or "non drop" in s_l:
            # Many panels indicate stability via non-drop wording.
            refill = "Non-Drop"
        else:
            # Common refill window marker (e.g., "30 Days")
            m2 = re.search(r"\b\d+\s*(?:day|days|week|weeks|month|months|year|years)\b", s, flags=re.IGNORECASE)
            refill = _pick(m2)

    # --- Start time ---
    start_time: Optional[str] = None
    m = re.search(r"(?:^|\b)(?:start(?:\s*time)?)\s*[:\-\–\—]\s*([^\n\r\|]+)", s, flags=re.IGNORECASE)
    start_time = _pick(m)
    if not start_time:
        m2 = re.search(r"\b\d+\s*-\s*\d+\s*(?:min|mins|minute|minutes|hr|hrs|hour|hours|day|days)\b", s, flags=re.IGNORECASE)
        start_time = _pick(m2)
    if not start_time and "instant start" in s_l:
        start_time = "Instant Start"

    return speed, refill, start_time


async def sync_catalog(session: AsyncSession) -> Dict[str, Any]:
    vendors = list((await session.execute(select(Vendor).where(Vendor.is_active == True).order_by(Vendor.id.asc()))).scalars().all())
    if not vendors:
        return {"vendors": 0, "services_seen": 0, "created": 0, "updated": 0, "vendor_maps": 0, "changes": []}

    adapters = [SMMv1Adapter(v) for v in vendors]
    results = await asyncio.gather(*[a.fetch_services() for a in adapters], return_exceptions=True)

    dtos: List[VendorServiceDTO] = []
    ok_vendors = 0
    for v, res in zip(vendors, results):
        v.last_sync_at = datetime.now(timezone.utc)
        if isinstance(res, Exception):
            v.last_sync_ok = False
            v.last_error = str(res)[:500]
            continue
        v.last_sync_ok = True
        v.last_error = None
        v.last_services_count = len(res)
        ok_vendors += 1
        dtos.extend(res)

    await commit_with_retry(session)

    # PostgreSQL-native upserts
    dialect_insert = pg_insert

    # --- Performance note ---
    # The old implementation did per-row lookups and per-row upserts, which becomes
    # extremely slow on PostgreSQL for large catalogs. This implementation performs:
    # 1) One bulk upsert into services (RETURNING ids)
    # 2) One (chunked) bulk upsert into vendor_services
    # 3) One bulk "mark inactive" update

    grouped: Dict[Tuple[str, str, str, str], List[VendorServiceDTO]] = {}
    for dto in dtos:
        k = _svc_key(dto.platform, dto.name, dto.type, dto.sub_type)
        grouped.setdefault(k, []).append(dto)

    logger.info(f"Grouped {len(dtos)} raw items into {len(grouped)} service groups.")

    # Snapshot existing services once (to compute created/updated counts without
    # performing per-service SELECTs).
    existing_rows = (
        await session.execute(
            select(
                Service.id,
                Service.platform,
                Service.name,
                Service.type,
                func.coalesce(Service.sub_type, ""),
                Service.category,
                Service.rate,
                Service.min,
                Service.max,
                Service.speed,
                Service.refill,
                Service.start_time,
                Service.is_active,
            )
        )
    ).all()

    existing_map: Dict[Tuple[str, str, str, str], Tuple[int, Tuple[Any, ...]]] = {}
    for (
        sid,
        platform,
        name,
        type_,
        sub_type,
        category,
        rate,
        min_,
        max_,
        speed,
        refill,
        start_time,
        is_active,
    ) in existing_rows:
        key = _svc_key(platform or "", name or "", type_ or "", sub_type or "")
        sig = (
            float(rate or 0.0),
            int(min_ or 0),
            int(max_ or 0),
            category,
            (sub_type or ""),
            (speed or None),
            (refill or None),
            (start_time or None),
            bool(is_active),
        )
        existing_map[key] = (int(sid), sig)

    services_payload: List[Dict[str, Any]] = []
    service_key_to_payload_idx: Dict[Tuple[str, str, str, str], int] = {}

    for k, items in grouped.items():
        cheapest = min(items, key=lambda x: x.vendor_rate)

        platform_n = (cheapest.platform or "").strip()
        name_n = (cheapest.name or "").strip()
        type_n = (cheapest.type or "").strip()
        sub_type_n = (cheapest.sub_type or "").strip()
        category_n = (cheapest.category or "").strip()

        # Extract optional service metadata from vendor description (best effort).
        desc_src = (cheapest.description or "").strip()
        if not desc_src:
            for it in items:
                if (it.description or "").strip():
                    desc_src = (it.description or "").strip()
                    break
        speed, refill, start_time = _extract_specs_from_description(desc_src)

        norm_key = _svc_key(platform_n, name_n, type_n, sub_type_n)

        payload = {
            "platform": platform_n,
            "category": category_n,
            "name": name_n,
            "type": type_n,
            "sub_type": sub_type_n,
            "rate": _to_float(cheapest.vendor_rate),
            "min": _to_int(cheapest.vendor_min),
            "max": _to_int(cheapest.vendor_max),
            "speed": speed,
            "refill": refill,
            "start_time": start_time,
            "is_active": True,
        }
        service_key_to_payload_idx[norm_key] = len(services_payload)
        services_payload.append(payload)

    created = 0
    updated = 0
    changes: List[Tuple[str, str, str]] = []

    
    # Bulk upsert services; RETURNING ids so we can upsert vendor mappings.
    # IMPORTANT: Chunk the VALUES to avoid huge parameter counts on PostgreSQL.
    returned: List[Tuple[Any, ...]] = []
    if services_payload:
        # Other DBs can handle larger chunks.
        logger.info(f"Prepared {len(services_payload)} services for insertion.")
        SVC_CHUNK = 1500
        for i in range(0, len(services_payload), SVC_CHUNK):
            chunk = services_payload[i : i + SVC_CHUNK]
            svc_upsert = dialect_insert(Service).values(chunk)
            svc_upsert = (
                svc_upsert.on_conflict_do_update(
                    index_elements=[Service.platform, Service.name, Service.type, Service.sub_type],
                    set_={
                        "category": svc_upsert.excluded.category,
                        "rate": svc_upsert.excluded.rate,
                        "min": svc_upsert.excluded.min,
                        "max": svc_upsert.excluded.max,
                        "speed": svc_upsert.excluded.speed,
                        "refill": svc_upsert.excluded.refill,
                        "start_time": svc_upsert.excluded.start_time,
                        "is_active": True,
                    },
                )
                .returning(
                    Service.id,
                    Service.platform,
                    Service.name,
                    Service.type,
                    func.coalesce(Service.sub_type, ""),
                )
            )
            returned.extend((await session.execute(svc_upsert)).all())

    # Build key->service_id map from RETURNING.
    service_id_by_key: Dict[Tuple[str, str, str, str], int] = {}
    seen_service_ids: List[int] = []
    for sid, platform, name, type_, sub_type in returned:
        key = _svc_key(platform or "", name or "", type_ or "", sub_type or "")
        service_id_by_key[key] = int(sid)
        seen_service_ids.append(int(sid))

        # created/updated accounting
        payload_idx = service_key_to_payload_idx.get(key)
        if payload_idx is None:
            continue
        p = services_payload[payload_idx]
        old = existing_map.get(key)
        new_sig = (
            float(p["rate"]),
            int(p["min"]),
            int(p["max"]),
            p["category"],
            p["sub_type"],
            (p.get("speed") or None),
            (p.get("refill") or None),
            (p.get("start_time") or None),
            True,
        )
        if not old:
            created += 1
            changes.append((p["platform"], p["category"], p["name"]))
        else:
            if old[1] != new_sig:
                updated += 1
                changes.append((p["platform"], p["category"], p["name"]))

    # Bulk upsert vendor mappings (chunked to avoid huge single statements)
    vendor_payload: List[Dict[str, Any]] = []
    for k, items in grouped.items():
        # We must use normalized key that matches RETURNING keys
        # (service key derived from cheapest fields).
        cheapest = min(items, key=lambda x: x.vendor_rate)
        platform_n = (cheapest.platform or "").strip()
        name_n = (cheapest.name or "").strip()
        type_n = (cheapest.type or "").strip()
        sub_type_n = (cheapest.sub_type or "").strip()
        key = _svc_key(platform_n, name_n, type_n, sub_type_n)
        service_id = service_id_by_key.get(key)
        if not service_id:
            continue
        for dto in items:
            vendor_payload.append(
                {
                    "service_id": service_id,
                    "vendor_id": int(dto.vendor_id),
                    "vendor_service_id": str(dto.vendor_service_id),
                    "vendor_rate": _to_float(dto.vendor_rate),
                    "vendor_min": _to_int(dto.vendor_min),
                    "vendor_max": _to_int(dto.vendor_max),
                }
            )

    vendor_maps = 0
    if vendor_payload:
        # Strict batching: never upsert the whole list at once.
        service_list = vendor_payload

        logger.info(f"Prepared {len(vendor_payload)} vendor mappings for insertion.")

        CHUNK_SIZE = 500
        for i in range(0, len(service_list), CHUNK_SIZE):
            chunk = service_list[i : i + CHUNK_SIZE]
            stmt = dialect_insert(VendorService).values(chunk)
            stmt = stmt.on_conflict_do_nothing(index_elements=["service_id", "vendor_id"])
            await session.execute(stmt)
            vendor_maps += len(chunk)

    await commit_with_retry(session)

    # Mark not-seen services inactive only if at least one vendor succeeded.
    # SAFETY: If a vendor/API glitch returns a partial catalog, we MUST NOT "soft wipe"
    # visibility by flipping most services to inactive.
    if ok_vendors > 0:
        MIN_SEEN_FOR_DEACTIVATE = 50
        seen_count = len(seen_service_ids or [])
        if seen_count < MIN_SEEN_FOR_DEACTIVATE:
            logger.warning(
                "Sync aborted (too few services returned: %s < %s), keeping existing services active.",
                seen_count,
                MIN_SEEN_FOR_DEACTIVATE,
            )
        else:
            await session.execute(update(Service).where(~Service.id.in_(seen_service_ids)).values(is_active=False))
            await commit_with_retry(session)

    return {
        "vendors": len(vendors),
        "vendors_ok": ok_vendors,
        "services_seen": len(dtos),
        "services_grouped": len(grouped),
        "created": created,
        "updated": updated,
        "vendor_maps": vendor_maps,
        "changes": changes[:60],  # cap for UI
    }


# -----------------------------
# Routing / ordering
# -----------------------------
async def pick_cheapest_vendor(session: AsyncSession, service_id: int) -> Tuple[VendorService, Vendor]:
    stmt = (
        select(VendorService, Vendor)
        .join(Vendor, Vendor.id == VendorService.vendor_id)
        .where(VendorService.service_id == service_id, Vendor.is_active == True)
        .order_by(VendorService.vendor_rate.asc())
    )
    row = (await session.execute(stmt)).first()
    if not row:
        raise RuntimeError("No active vendor mapping found for this service.")
    return row[0], row[1]


def _vendor_adapter(vendor: Vendor) -> SMMv1Adapter:
    return SMMv1Adapter(vendor)


async def create_order(
    session: AsyncSession,
    user: User,
    service: Service,
    link: str,
    quantity: int,
    markup_percent: float,
) -> Order:
    # Fetch top 3 cheapest active vendors for this service (failover candidates)
    stmt = (
        select(VendorService, Vendor)
        .join(Vendor, Vendor.id == VendorService.vendor_id)
        .where(VendorService.service_id == service.id, Vendor.is_active == True)
        .order_by(VendorService.vendor_rate.asc())
        .limit(3)
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        raise RuntimeError("No active vendor mapping found for this service.")

    # Price/charge is computed from the cheapest vendor mapping; failover may reduce profit.
    vs0, vendor0 = rows[0][0], rows[0][1]

    if quantity < int(vs0.vendor_min) or quantity > int(vs0.vendor_max):
        raise RuntimeError(f"Quantity must be between {vs0.vendor_min} and {vs0.vendor_max} for this service.")

    effective_markup = float(service.custom_markup) if service.custom_markup is not None else float(markup_percent)
    display_rate = apply_markup(float(vs0.vendor_rate), effective_markup)
    charge = round(calc_charge(display_rate, quantity), 4)

    if float(user.balance) < charge:
        raise RuntimeError(f"Insufficient balance. Need {charge:.4f}, you have {float(user.balance):.4f}")

    # Deduct user balance once up-front. Only refund if ALL vendors fail.
    user.balance = float(user.balance) - charge
    await session.flush()

    # Create the order record once; vendor fields will be updated when a vendor succeeds.
    # Use the first candidate as the initial vendor_name for transparency.
    order = Order(
        user_id=user.id,
        service_id=service.id,
        link=link,
        quantity=quantity,
        charge=charge,
        cost=0.0,
        profit=0.0,
        status="Processing",
        vendor_name=vendor0.name,
        vendor_order_id=None,
    )
    session.add(order)
    await session.flush()

    last_error: Optional[str] = None

    for vs, vendor in [(r[0], r[1]) for r in rows]:
        # Skip vendor mappings that cannot support requested quantity
        try:
            if quantity < int(vs.vendor_min) or quantity > int(vs.vendor_max):
                logger.warning(
                    "Vendor %s mapping skipped (qty %s not in %s-%s)",
                    vendor.name,
                    quantity,
                    vs.vendor_min,
                    vs.vendor_max,
                )
                continue
        except Exception as e:
            # If vendor_min/max are malformed, try anyway and let vendor reject
            logger.exception(f"Error processing order: {e}")

        # Update financials for THIS vendor attempt (charge stays fixed)
        try:
            cost = round((float(vs.vendor_rate) / 1000.0) * float(quantity), 4)
        except Exception:
            cost = 0.0
        profit = round(float(charge) - float(cost), 4)

        adapter = _vendor_adapter(vendor)
        try:
            vendor_order_id = await adapter.add_order(vs.vendor_service_id, link=link, quantity=quantity)

            order.vendor_name = vendor.name
            order.vendor_order_id = vendor_order_id
            order.cost = cost
            order.profit = profit

            await commit_with_retry(session)
            await session.refresh(order)
            return order
        except Exception as e:
            last_error = str(e)
            logger.warning("Vendor add_order failed for %s; trying next vendor. Error: %s", vendor.name, last_error)

            # Do not refund here. Continue to next vendor.
            continue

    # All vendors failed -> refund and mark order failed
    logger.error("All vendor attempts failed; refunding user. Last error: %s", last_error)
    user.balance = float(user.balance) + charge
    order.status = "Failed"
    await commit_with_retry(session)

    raise RuntimeError(f"All vendors failed for this order. Last error: {last_error or 'unknown'}")


# -----------------------------
# Crypto Pay client
# -----------------------------
class CryptoPayClient:
    def __init__(self, api_token: str, base_url: str = "https://pay.crypt.bot/api") -> None:
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")

    async def _call(self, method: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{method}"
        headers = {"Crypto-Pay-API-Token": self.api_token}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, headers=headers, json=(payload or {}))
            r.raise_for_status()
            data = r.json()
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise RuntimeError(f"CryptoPay {method} failed: {data!r}")
        return data["result"]

    async def create_invoice(self, amount: float, asset: str, description: str, payload: str, expires_in: int = 3600) -> Dict[str, Any]:
        return await self._call(
            "createInvoice",
            {
                "asset": asset,
                "amount": f"{amount:.2f}",
                "description": description,
                "payload": payload,
                "expires_in": expires_in,
                "allow_comments": False,
                "allow_anonymous": True,
            },
        )

    async def get_invoice(self, invoice_id: int) -> Dict[str, Any]:
        invoices = await self._call("getInvoices", {"invoice_ids": str(invoice_id)})
        if isinstance(invoices, list) and invoices:
            return invoices[0]
        raise RuntimeError(f"Invoice not found: {invoice_id}")





# -----------------------------
# Crypto Pay deposit polling
# -----------------------------




def normalize_vendor_status(raw_status: str) -> str:
    """Normalize raw vendor status strings into canonical bot statuses.

    Vendors return inconsistent status strings (case, spacing, synonyms). We normalize
    them so downstream logic can reliably compare against canonical values like:
      - "Completed"
      - "Processing"
      - "Pending"
      - "Cancelled"
      - "Partial"
      - "Refunded"

    Important: some panels return API *errors* in the status field (e.g. "Error").
    Those are not real order states; we treat them as "Processing" so we don't
    incorrectly downgrade an in-flight order to "Failed".
    """
    try:
        s = str(raw_status or "")
    except Exception:
        s = ""
    s = s.strip().lower()
    if not s:
        return "Processing"

    # Collapse whitespace (some panels return "In   Progress", etc.)
    try:
        s = re.sub(r"\s+", " ", s)
    except Exception:
        pass

    # API errors sometimes show up as statuses; keep the order in-flight.
    api_error_vals = {"error", "unknown", "n/a", "na", "none", "null"}
    if s in api_error_vals or s.startswith("error:") or s.startswith("err:"):
        return "Processing"

    completed_vals = {
        "completed",
        "complete",
        "done",
        "success",
        "successful",
        "delivered",
        "finished",
        "ok",
    }
    cancelled_vals = {
        "cancelled",
        "canceled",
        "cancel",
        "canceled by user",
        "cancelled by user",
        "canceled by client",
        "cancelled by client",
    }
    processing_vals = {
        "processing",
        "in progress",
        "progress",
        "running",
        "working",
        "active",
    }
    pending_vals = {
        "pending",
        "awaiting",
        "waiting",
        "queued",
        "queue",
        "new",
    }
    partial_vals = {
        "partial",
        "partially",
        "partially completed",
        "partial completed",
    }
    # "Failed" is not a standard SMM order state; keep this conservative.
    failed_vals = {
        "failed",
        "rejected",
        "declined",
    }
    refunded_vals = {
        "refunded",
        "refund",
        "returned",
    }

    if s in completed_vals:
        return "Completed"
    if s in cancelled_vals:
        return "Cancelled"
    if s in processing_vals:
        return "Processing"
    if s in pending_vals:
        return "Pending"
    if s in partial_vals:
        return "Partial"
    if s in failed_vals:
        return "Failed"
    if s in refunded_vals:
        return "Refunded"

    # Keyword-based fallbacks
    if "complete" in s or "success" in s or "deliver" in s or "finish" in s:
        return "Completed"
    if "cancel" in s:
        return "Cancelled"
    if "pend" in s or "queue" in s:
        return "Pending"
    if "process" in s or "progress" in s:
        return "Processing"
    if "partial" in s:
        return "Partial"
    if "refund" in s:
        return "Refunded"
    # Avoid downgrading on generic error-like strings.
    if "error" in s or "invalid" in s or "not found" in s:
        return "Processing"
    if "fail" in s or "reject" in s or "declin" in s:
        return "Failed"

    # Default: title-case for readability (keeps bot status stable)
    try:
        return s.title()
    except Exception:
        return "Processing"



async def poll_processing_orders(session: AsyncSession, app: Optional[Application] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Poll vendor statuses for in-flight orders.

    Updates Order.status for orders in Pending/Processing, edits receipts (best effort),
    and returns a list of completion notifications for the caller to send.

    NOTE: This function does NOT commit. The caller owns the transaction boundary.
    """
    await ensure_order_notifications_table(session)

    for attempt in range(3):
        try:
            stmt = (
                select(Order, User.telegram_id, Service, Vendor)
                .join(User, Order.user_id == User.id)
                .outerjoin(Service, Order.service_id == Service.id)
                .outerjoin(Vendor, Vendor.name == Order.vendor_name)
                .outerjoin(
                    VendorService,
                    (VendorService.service_id == Service.id) & (VendorService.vendor_id == Vendor.id),
                )
                .where(
                    Order.vendor_order_id.is_not(None),
                    Order.status.in_(["Pending", "Processing"]),
                )
                .order_by(Order.created_at.asc())
                .limit(int(limit))
            )

            rows = (await session.execute(stmt)).all()
            notifications: List[Dict[str, Any]] = []

            for order, telegram_id, service, vendor in rows:
                try:
                    # Ensure we don't get stuck on orphaned orders (missing service/vendor rows)
                    if service is None:
                        order.status = "Error - Service Missing"
                        continue

                    service_name = getattr(service, "name", None) or "Unknown Service"
                    service_platform = getattr(service, "platform", None) or "Unknown"
                    service_category = getattr(service, "category", None) or "Unknown"

                    if not vendor:
                        order.status = "Error - Vendor Missing"
                        continue

                    if not bool(getattr(vendor, "is_active", True)):
                        order.status = "Error - Vendor Inactive"
                        continue

                    adapter = _vendor_adapter(vendor)
                    raw_status = await adapter.get_order_status(str(order.vendor_order_id))
                    raw_status_str = str(raw_status or "")
                    new_status = normalize_vendor_status(raw_status_str)

                    # Debug visibility: log raw vendor response + parsed status.
                    try:
                        logger.info(
                            "Order status poll: bot_order_id=%s vendor=%s vendor_order_id=%s raw_status=%r parsed_status=%s",
                            int(order.id),
                            str(getattr(vendor, "name", "") or ""),
                            str(order.vendor_order_id),
                            raw_status_str,
                            new_status,
                        )
                    except Exception:
                        logger.exception("Failed to log order status poll details")

                    # Conservative handling: don't downgrade in-flight orders to Failed due to vendor/API quirks.
                    if new_status == "Failed" and str(order.status) in ("Pending", "Processing"):
                        logger.warning(
                            "Ignoring vendor 'Failed' status for in-flight order: bot_order_id=%s vendor=%s vendor_order_id=%s raw_status=%r",
                            int(order.id),
                            str(getattr(vendor, "name", "") or ""),
                            str(order.vendor_order_id),
                            raw_status_str,
                        )
                        continue

                    # Apply status update if it changed.
                    if order.status != new_status:
                        order.status = new_status

                    # Completion flow (runs even if status didn't change, so missed notifications can recover).
                    if new_status == "Completed":
                        # 1) Edit receipt (best effort)
                        if app and order.receipt_message_id and order.receipt_chat_id:
                            try:
                                receipt_text = (
                                    "🧾 <b>Order Receipt</b>\n\n"
                                    f"🆔 <b>Bot Order ID:</b> <code>{_html.escape(str(order.id))}</code>\n"
                                    f"📦 <b>Service:</b> {_html.escape(str(service_name))}\n"
                                    f"🏷️ <b>Platform/Category:</b> {_html.escape(str(service_platform))} • {_html.escape(str(service_category))}\n"
                                    f"🔗 <b>Link:</b> {_html.escape(str(order.link))}\n"
                                    f"🔢 <b>Quantity:</b> {_html.escape(str(order.quantity))}\n"
                                    f"💳 <b>Total:</b> {_html.escape((f'${float(order.charge):.4f}').rstrip('0').rstrip('.'))}\n"
                                    f"🌐 <b>Order ID:</b> <code>{_html.escape(str(order.vendor_order_id))}</code>"
                                )
                                await app.bot.edit_message_text(
                                    chat_id=order.receipt_chat_id,
                                    message_id=order.receipt_message_id,
                                    text=receipt_text,
                                    parse_mode="HTML",
                                    disable_web_page_preview=True,
                                )
                            except Exception as e:
                                logger.exception(f"Error processing order: {e}")

                        # 2) Ensure we only notify once
                        res = (
                            await session.execute(
                                text("SELECT completed_notified FROM order_notifications WHERE order_id = :oid"),
                                {"oid": int(order.id)},
                            )
                        ).first()

                        already = bool(res and int(res[0]) == 1)
                        if not already:
                            await session.execute(
                                text(
                                    "INSERT INTO order_notifications (order_id, completed_notified, created_at) "
                                    "VALUES (:oid, 1, :ts) "
                                    "ON CONFLICT(order_id) DO UPDATE SET completed_notified=1"
                                ),
                                {"oid": int(order.id), "ts": datetime.now(timezone.utc).isoformat()},
                            )

                            notifications.append(
                                {
                                    "telegram_id": int(telegram_id),
                                    "order_id": int(order.id),
                                    "service_name": str(service_name),
                                    "quantity": int(order.quantity),
                                    "vendor_order_id": str(order.vendor_order_id),
                                    "link": str(order.link) if order.link else None,
                                }
                            )


                except Exception as e:
                    logger.exception(f"Error processing order: {e}")

            return notifications

        except OperationalError as e:
            if attempt < 2:
                try:
                    await session.rollback()
                except Exception as e2:
                    logger.exception(f"Error processing order: {e2}")
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            raise


async def create_deposit_transaction(session: AsyncSession, user: User, amount: float, invoice_id: int, asset: str, pay_url: str, payload: Optional[str] = None) -> Transaction:
    for _attempt in range(3):
        try:
            # Store a JSON payload so UI can show invoice_id/asset/pay_url (fixes 'invoice ?').
            # Keep the raw payload string inside JSON so reverse-polling can still match by LIKE '%payload%'.
            payload_str = payload or f"tg:{getattr(user, 'telegram_id', user.id)}:{uuid.uuid4()}"
            try:
                inv_int: Any = int(invoice_id)
            except Exception:
                inv_int = str(invoice_id)

            payment_obj = {
                "invoice_id": inv_int,
                "asset": (asset or "").upper(),
                "pay_url": pay_url or "",
                "payload": payload_str,
            }
            tx = Transaction(
                user_id=user.id,
                amount=float(amount),
                status="pending",
                payment_payload=json.dumps(payment_obj, ensure_ascii=False),
            )
            session.add(tx)
            await session.flush()
            await session.refresh(tx)
            await session.commit()
            return tx
        except OperationalError as e:
            if _attempt < 2:
                try:
                    await session.rollback()
                except Exception as e:
                    logger.exception(f"Error processing order: {e}")
                await asyncio.sleep(0.5)
                continue
            raise
async def check_crypto_deposits(session: AsyncSession, *args, limit: int = 200) -> int:
    """Crypto Pay deposit polling (reverse polling, payload matching).

    Supports calls:
      - await check_crypto_deposits(session, crypto, limit=200)
      - await check_crypto_deposits(session, app, crypto, limit=200)

    Flow:
      1) Fetch recent PAID invoices from Crypto Pay API (status='paid') (API FIRST).
      2) For each paid invoice, extract payload string.
      3) Find local Transaction where status='pending' and payment_payload contains payload.
      4) Atomically mark tx paid, credit user's balance, and send Telegram notification (best-effort).
    """
    # Parse args
    app = None
    crypto = None
    if len(args) == 1:
        crypto = args[0]
    elif len(args) >= 2:
        app = args[0]
        crypto = args[1]
    if crypto is None:
        raise TypeError("check_crypto_deposits() missing required argument: 'crypto'")

    updated_count = 0

    # 1) Fetch paid invoices from API first.
    try:
        count = int(limit)
        if count <= 0:
            count = 100
        if count > 200:
            count = 200

        result = await crypto._call("getInvoices", {"status": "paid", "offset": 0, "count": count})

        if isinstance(result, dict) and "items" in result and isinstance(result["items"], list):
            paid_invoices = result["items"]
        elif isinstance(result, list):
            paid_invoices = result
        else:
            paid_invoices = []
    except Exception as e:
        logger.exception("Crypto Pay reverse poller: failed to fetch paid invoices: %s", e)
        return 0

    if not paid_invoices:
        return 0

    # 2) Match by payload -> pending tx.
    for inv in paid_invoices:
        try:
            payload = str(inv.get("payload") or "").strip()
        except Exception:
            payload = ""

        if not payload:
            continue

        inv_status = str(inv.get("status", "") or "").strip().lower()
        if inv_status not in ("paid", "settled", "completed", "success", "successful"):
            continue

        try:
            tx = (
                await session.execute(
                    select(Transaction)
                    .where(Transaction.status == "pending")
                    .where(Transaction.payment_payload.like(f"%{payload}%"))
                    .order_by(desc(Transaction.created_at))
                    .limit(1)
                )
            ).scalars().first()
        except Exception as e:
            logger.exception("Crypto Pay reverse poller: DB lookup failed for payload %s: %s", payload, e)
            continue

        if not tx:
            continue

        # 3) Atomic status-guard update to prevent double-crediting.
        try:
            res = await session.execute(
                update(Transaction)
                .where(Transaction.id == tx.id)
                .where(Transaction.status == "pending")
                .values(status="completed")
            )
            if not getattr(res, "rowcount", 0):
                try:
                    await session.rollback()
                except Exception as e:
                    logger.exception(f"Error processing order: {e}")
                continue

            usr = (await session.execute(select(User).where(User.id == tx.user_id))).scalars().first()
            if usr:
                usr.balance = float(usr.balance) + float(tx.amount)

            await commit_with_retry(session)
            updated_count += 1

            # 4) Best-effort Telegram notification (must not crash poller).
            try:
                asset = ""
                if isinstance(tx.payment_payload, str) and tx.payment_payload:
                    parts = tx.payment_payload.split(":")
                    if len(parts) >= 4:
                        # format: tg:user_id:asset:{ASSET}:usd:{amount}
                        # or tg:user_id:asset:{ASSET}:usd:{amount}
                        if parts[2].lower() == "asset":
                            asset = parts[3].upper()
                        else:
                            asset = parts[2].upper()
                invoice_id = inv.get("invoice_id") or inv.get("id") or inv.get("invoiceId") or ""
                try:
                    invoice_id = int(invoice_id)
                except Exception as e:
                    logger.exception(f"Error processing order: {e}")

                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_balance = ""
                try:
                    new_balance = f"{float(getattr(usr, 'balance', 0.0)):.2f}"
                except Exception:
                    new_balance = str(getattr(usr, 'balance', ''))

                receipt_url = getattr(tx, "pay_url", None) or getattr(tx, "payment_url", None) or ""
                receipt_line = f"\n🔗 Receipt: {receipt_url}" if receipt_url else ""

                msg = (
                    "✅ Deposit Successfully Received\n"
                    f"💰 Amount: ${float(tx.amount):.2f} {asset}\n"
                    f"🆔 Invoice ID: {invoice_id}\n"
                    f"💳 New Balance: ${new_balance}\n"
                    f"📅 Date: {now_str}"
                    f"{receipt_line}"
                )
                # Prefer app.bot if available
                if app is not None and getattr(app, "bot", None) is not None and getattr(usr, "telegram_id", None):
                    await app.bot.send_message(chat_id=int(usr.telegram_id), text=msg)
                else:
                    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
                    chat_id = int(getattr(usr, "telegram_id", None) or tx.user_id)
                    if token:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            await client.post(
                                f"https://api.telegram.org/bot{token}/sendMessage",
                                data={"chat_id": chat_id, "text": msg},
                            )
                    else:
                        logger.warning("No BOT_TOKEN configured; skipping deposit notification to chat_id=%s", chat_id)
            except Exception as e:
                logger.warning("Failed to send deposit notification: %s", e)

        except Exception as e:
            logger.exception("Crypto Pay reverse poller: failed to credit tx %s: %s", getattr(tx, "id", None), e)
            try:
                await session.rollback()
            except Exception as e:
                logger.exception(f"Error processing order: {e}")
            continue

    return updated_count



# -----------------------------
# Global engine/session (importable)
# -----------------------------
# bot.py expects these at module level:
#   from database import engine, async_session
#
# IMPORTANT:
# - Do NOT instantiate full Settings() at import time (BOT_TOKEN should not be required just to import database.py).
# - Prefer DATABASE_URL (from env/.env). If missing, stay import-safe and require init_engine() at runtime.

load_dotenv()

_engine_url = os.getenv("DATABASE_URL")

engine = create_async_engine(_engine_url, echo=False, future=True) if _engine_url else None
async_session = (
    async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession) if engine else None
)

def init_engine(database_url: str | None = None):
    """Initialize (or return existing) async SQLAlchemy engine + sessionmaker."""
    global engine, async_session, _engine_url
    if engine is not None and async_session is not None:
        return engine, async_session

    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        # Don't crash on import; only raise when initialization is actually needed.
        raise RuntimeError("DATABASE_URL is not set; cannot initialize database engine")

    _engine_url = url
    engine = create_async_engine(url, echo=False, future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, async_session

