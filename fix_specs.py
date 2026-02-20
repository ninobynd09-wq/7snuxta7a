#!/usr/bin/env python3
"""
fix_specs.py

Backfill Service specs (speed/refill/start_time) from the `services.name` text.

Strict constraints:
- Never overwrite existing values.
- Only update rows where these columns are NULL/empty.

Usage:
  python3 fix_specs.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional, Tuple

from dotenv import load_dotenv
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Local project imports
from models import Service

try:
    from config import load_settings  # type: ignore
except Exception:  # pragma: no cover
    load_settings = None  # type: ignore


LOG = logging.getLogger("fix_specs")

# --- Regex helpers -----------------------------------------------------------

_RE_SPEED = re.compile(r"(?is)\bSpeed\s*[:\-]?\s*([^\)\]\|\n\r]+)")
_RE_START = re.compile(r"(?is)\bStart\s*[:\-]?\s*([^\)\]\|\n\r]+)")

# Refill patterns (much more robust)
_RE_DAYS_REFILL = re.compile(r"(?is)\b(\d{1,4})\s*(?:days?|d)\s*(?:refill|guaranteed)", re.I)
_RE_REFILL_DAYS = re.compile(r"(?is)refill\s*[:\-]?\s*(\d{1,4})\s*(?:days?|d)", re.I)

_RE_NON_DROP = re.compile(r"(?is)\bnon[\s\-]*drop\b")
_RE_NO_REFILL = re.compile(r"(?is)\bno[\s\-]*refill\b")

_RE_INSTANT = re.compile(r"(?is)\binstant\b")


def _clean(val: str) -> str:
    v = re.sub(r"\s+", " ", val).strip()
    v = v.strip(" -|,;")
    return v


def parse_specs_from_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (speed, refill, start_time) parsed from a service name string.
    """
    if not name:
        return None, None, None

    speed: Optional[str] = None
    refill: Optional[str] = None
    start_time: Optional[str] = None

    # Speed
    m = _RE_SPEED.search(name)
    if m:
        speed = _clean(m.group(1))

    # Refill - ordered by specificity (no generic "Refill" anymore)
    m = _RE_DAYS_REFILL.search(name)
    if m:
        refill = f"{m.group(1)} Days Refill"
    else:
        m = _RE_REFILL_DAYS.search(name)
        if m:
            refill = f"{m.group(1)} Days Refill"
        elif _RE_NON_DROP.search(name):
            refill = "Non Drop"
        elif _RE_NO_REFILL.search(name):
            refill = "No Refill"

    # Start time
    m = _RE_START.search(name)
    if m:
        start_time = _clean(m.group(1))
    elif _RE_INSTANT.search(name):
        start_time = "Instant"

    return speed, refill, start_time


def _is_empty(v: Optional[str]) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


async def run() -> None:
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    if load_settings is not None:
        try:
            s = load_settings()
            if getattr(s, "database_url", None):
                db_url = s.database_url
        except Exception:
            pass

    if not db_url:
        raise SystemExit("DATABASE_URL is not set (and config.py could not provide it).")

    engine = create_async_engine(db_url, echo=False, future=True)
    Session: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    updated = 0
    scanned = 0

    async with Session() as session:
        stmt = select(Service).where(or_(Service.speed.is_(None), Service.speed == ""))
        stream = await session.stream_scalars(stmt)

        pending = 0
        async for svc in stream:
            scanned += 1

            speed_p, refill_p, start_p = parse_specs_from_name(svc.name or "")

            changed = False
            if _is_empty(svc.speed) and speed_p:
                svc.speed = speed_p
                changed = True
            if _is_empty(svc.refill) and refill_p:
                svc.refill = refill_p
                changed = True
            if _is_empty(svc.start_time) and start_p:
                svc.start_time = start_p
                changed = True

            if changed:
                updated += 1
                pending += 1

            if pending >= 500:
                await session.commit()
                pending = 0

        if pending:
            await session.commit()

    await engine.dispose()

    LOG.info("Done. Scanned=%s Updated=%s", scanned, updated)
    print(f"Done. Scanned={scanned} Updated={updated}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
