from __future__ import annotations

import os
import sys
# Ensure local imports work even if the bot is started from a different working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import uuid

import asyncio
import html
import html as _html
import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from urllib.parse import urlparse

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


async def _safe_answer(q) -> None:
    """Answer callback queries without crashing on network timeouts."""
    try:
        await q.answer()
    except telegram.error.TimedOut:
        # Ignore timeout and continue handler logic
        pass


async def _safe_delete_message(bot, chat_id: int, message_id: int) -> None:
    """Best-effort message deletion (never crash the flow)."""
    try:
        if not chat_id or not message_id:
            return
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except (telegram.error.BadRequest, telegram.error.Forbidden, telegram.error.TimedOut):
        return
    except Exception:
        return


async def _cleanup_temp_order_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete ONLY the transient messages we explicitly track (detail/link/qty + user inputs)."""
    try:
        chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    except Exception:
        chat_id = 0
    if not chat_id:
        return

    # Do NOT touch the Service List message or the Order Summary message.
    for key in ("detail_msg_id", "user_link_msg_id", "qty_request_msg_id", "user_qty_msg_id"):
        mid = context.user_data.get(key)
        try:
            mid_i = int(mid) if mid else 0
        except Exception:
            mid_i = 0
        if mid_i:
            await _safe_delete_message(context.bot, chat_id, mid_i)
        # Clear after attempt
        context.user_data.pop(key, None)



async def _nuclear_wipe_order_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete ALL tracked transient order-flow messages and reset tracking keys.

    Intended for Cancel / hard failure paths to avoid 'soft' leftovers.
    Does NOT touch the Service List menu messages.
    """
    try:
        chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    except Exception:
        chat_id = 0
    if not chat_id:
        return

    # Collect message ids first (avoid deleting the same id twice)
    mids: set[int] = set()

    # Callback message (often the Order Summary) should be deleted on Cancel/failure
    try:
        q = update.callback_query
        if q and q.message and getattr(q.message, "message_id", None):
            mids.add(int(q.message.message_id))
    except Exception:
        pass

    for key in (
        "detail_msg_id",
        "user_link_msg_id",
        "qty_request_msg_id",
        "user_qty_msg_id",
        # summary/order-confirm message id (same thing in this project)
        "summary_msg_id",
        "order_confirm_message_id",
        # validation error message (e.g., minimum order)
        "error_msg_id",
    ):
        mid = context.user_data.get(key)
        try:
            mid_i = int(mid) if mid else 0
        except Exception:
            mid_i = 0
        if mid_i:
            mids.add(mid_i)
        context.user_data.pop(key, None)

    for mid_i in mids:
        await _safe_delete_message(context.bot, chat_id, mid_i)


def _build_order_receipt_text(svc: Service, charge: float, qty: int, status: str) -> str:
    """Build a clean, consistent Order Receipt (plain text) that we can also edit later."""
    # Base service name
    svc_name = (getattr(svc, "name", "") or "").strip()
    platform = (getattr(svc, "platform", "") or "").strip()
    base = f"{platform} - {svc_name}" if platform else svc_name

    tokens: list[str] = []

    variant = (getattr(svc, "sub_type", "") or "").strip()
    if variant:
        tokens.append(f"[Variant: {variant}]")

    # Refill display: prefer explicit value, otherwise 'No'
    refill_raw = (getattr(svc, "refill", "") or "").strip()
    if not refill_raw:
        refill_disp = "No"
    else:
        refill_disp = "No" if "no" in refill_raw.lower() else refill_raw
    tokens.append(f"[Refill: {refill_disp}]")

    # Min/Max (always useful in receipt)
    try:
        min_v = int(getattr(svc, "min", 0) or 0)
    except Exception:
        min_v = 0
    try:
        max_v = int(getattr(svc, "max", 0) or 0)
    except Exception:
        max_v = 0

    if min_v:
        tokens.append(f"[Min: {min_v}]")
    if max_v:
        tokens.append(f"[Max: {max_v}]")

    # Optional specs
    speed = (getattr(svc, "speed", "") or "").strip()
    start_time = (getattr(svc, "start_time", "") or "").strip()
    if speed:
        tokens.append(f"[Speed: {speed}]")
    if start_time:
        tokens.append(f"[Start: {start_time}]")

    service_line = base
    if tokens:
        service_line = f"{base} " + " ".join(tokens)

    try:
        charge_f = float(charge)
    except Exception:
        charge_f = 0.0

    status_clean = (status or "").strip() or "Pending"

    return "\n".join(
        [
            "🧾 Order Receipt",
            f"Service: {service_line}",
            f"Cost: ${charge_f:.2f} | Qty: {int(qty)}",
            f"Status: {status_clean}",
        ]
    )

from config import load_settings
from models import Base, Service, Order, Transaction, Vendor, User, MenuVisibility
from database import (
    CryptoPayClient,
    SMMv1Adapter,
    PLATFORM_LIST,
    engine,
    async_session,
    init_engine,
    add_funds,
    add_vendor,
    apply_markup,
    bootstrap_from_env,
    calc_charge,
    commit_with_retry,
    create_deposit_transaction,
    create_order,
    get_markup_percent,
    get_or_create_user,
    is_maintenance_on,
    toggle_maintenance,
    list_all_vendors,
    set_markup_percent,
    sync_catalog,
    test_vendor,
    toggle_vendor,
    pick_cheapest_vendor,
    poll_processing_orders,
    check_crypto_deposits,
)
from sqlalchemy import select, func, desc, text
from sqlalchemy.exc import OperationalError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("smm_bot")


# -----------------------------
# Navigation helpers / constants
# -----------------------------
MAIN = "main"
PLATFORMS = "platforms"
PLAT_CATEGORIES = "platform_categories"
PLAT_SUBTYPES = "platform_subtypes"
SERVICE_LIST = "service_list"

WALLET = "wallet"
DEPOSIT = "deposit"
TX = "tx"
ORDERS = "orders"
ORDER_DETAIL = "order_detail"

SUPPORT = "support"

# ordering input screens (for Back navigation)
ORDER_LINK = "order_link"
ORDER_QTY = "order_qty"
ORDER_CONFIRM = "order_confirm"

ADMIN = "admin"
ADMIN_VENDORS = "admin_vendors"
ADMIN_VENDOR_PICK = "admin_vendor_pick"
ADMIN_VENDOR_STATS = "admin_vendor_stats"
ADMIN_MENU_MANAGER = "admin_menu_manager"
ADMIN_DAILY_STATS = "admin_daily_stats"
ADMIN_SYNC_RESULT = "admin_sync_result"

# input states
INP_NONE = None
INP_ORDER_LINK = "order_link"
INP_ORDER_QTY = "order_qty"
INP_DEPOSIT_CUSTOM = "deposit_custom"
INP_MARKUP = "markup"
INP_ADDV_NAME = "addv_name"
INP_ADDV_URL = "addv_url"
INP_ADDV_KEY = "addv_key"
INP_ADDFUNDS_UID = "addfunds_uid"
INP_ADDFUNDS_AMT = "addfunds_amt"
INP_SVCMARKUP_SEARCH = "svcmarkup_search"
INP_SVCMARKUP_PCT = "svcmarkup_pct"
INP_SVCMARKUP_SUBTYPE = "svcmarkup_subtype"


INP_CHECK_ORDER = "check_order"

# -----------------------------
# Order conversation states
# -----------------------------
ORDER_CONV_LINK = 6101
ORDER_CONV_QTY = 6102
ORDER_CONV_CONFIRM = 6103
def money(x: float) -> str:
    return f"${x:.4f}".rstrip("0").rstrip(".")


def _short(s: str, n: int = 28) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _plat_short(platform: str) -> str:
    p = (platform or "").strip().lower()
    m = {
        "instagram": "Insta",
        "tiktok": "TikTok",
        "youtube": "YT",
        "x": "X",
        "twitter": "X",
        "x (twitter)": "X",
        "telegram": "TG",
        "facebook": "FB",
        "facebook page": "FB",
        "facebook pages": "FB",
    }
    if p in m:
        return m[p]
    # fallback: keep it short
    raw = (platform or "").strip()
    return raw if len(raw) <= 8 else raw[:8]


def _plat_display(platform: str) -> str:
    p = (platform or "").strip().lower()
    m = {
        "instagram": "Insta",
        "insta": "Insta",
        "tiktok": "TikTok",
        "tik tok": "TikTok",
        "youtube": "YT",
        "yt": "YT",
        "x": "X",
        "twitter": "X",
        "x (twitter)": "X",
        "telegram": "TG",
        "facebook": "FB",
        "facebook page": "FB",
        "facebook pages": "FB",
    }
    return m.get(p, (platform or "").strip())

def _service_flag(name: str) -> str:
    """Return a short region/quality indicator emoji for display (e.g., 🇺🇸, 🌍, 👤, 🛡️)."""
    raw = (name or "")
    low = raw.lower()

    # Collect tokens from bracket/paren tags plus free-text words
    tokens = set()
    for grp in re.findall(r"\[([^\]]+)\]", raw) + re.findall(r"\(([^)]+)\)", raw):
        for t in re.split(r"[|,/;]+", grp):
            t = t.strip().lower()
            if t:
                tokens.add(t)

    for t in re.findall(r"\b[a-zA-Z]{2,}\b", raw):
        tokens.add(t.lower())

    icons: List[str] = []

    def add_icon(icon: str) -> None:
        if icon and icon not in icons:
            icons.append(icon)

    # Country / region flags (match common tokens)
    if ("usa" in tokens) or ("u.s.a" in tokens) or ("united states" in low) or re.search(r"\b(us|u\.s\.)\b", low):
        add_icon("🇺🇸")
    elif ("uk" in tokens) or ("u.k" in tokens) or ("united kingdom" in low) or re.search(r"\b(uk|u\.k\.)\b", low):
        add_icon("🇬🇧")
    elif "france" in tokens:
        add_icon("🇫🇷")
    elif "germany" in tokens or "deutschland" in tokens:
        add_icon("🇩🇪")
    elif "canada" in tokens:
        add_icon("🇨🇦")
    elif "australia" in tokens:
        add_icon("🇦🇺")
    elif "india" in tokens:
        add_icon("🇮🇳")
    elif "turkey" in tokens:
        add_icon("🇹🇷")
    elif "brazil" in tokens:
        add_icon("🇧🇷")
    elif "spain" in tokens:
        add_icon("🇪🇸")
    elif "italy" in tokens:
        add_icon("🇮🇹")
    elif "russia" in tokens:
        add_icon("🇷🇺")
    elif "uae" in tokens or "emirates" in tokens or "united arab emirates" in low:
        add_icon("🇦🇪")

    # Global / Worldwide
    if ("global" in tokens) or ("worldwide" in tokens) or ("international" in tokens) or re.search(r"\b(global|worldwide|international)\b", low):
        add_icon("🌍")

    # Quality indicators often used as tags
    if re.search(r"\b(non[-\s]?drop|no[-\s]?drop)\b", low):
        add_icon("🛡️")
    if re.search(r"\b(real|hq|premium)\b", low):
        add_icon("👤")

    return " ".join(icons).strip()


def _clean_service_name(name: str) -> str:
    """Return a balanced, clean service name for button labels (keeps meaning via flags)."""
    s = (name or "").strip()
    if not s:
        return ""

    # Keep full raw for flag detection, but remove bracket/paren blocks from the displayed core name
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\([^)]*\)", "", s)

    # If the vendor packed multiple fields with pipes, keep the first segment (usually the core name)
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if parts:
        s = parts[0]

    # Remove noisy rate-limits like 50k/day, 10k day, etc.
    s = re.sub(r"\b\d+\s*[kK]?\s*/\s*day\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\s*[kK]?\s*day\b", "", s, flags=re.IGNORECASE)

    # Remove keywords and their trailing values where applicable
    s = re.sub(r"\b(speed|max|refill|start)\b\s*[:\-]?\s*[^\-–—|]*", "", s, flags=re.IGNORECASE)

    # Remove standalone noise words (but not country/region tags; those are handled via _service_flag)
    s = re.sub(r"\b(instant|new|random|day|days)\b", "", s, flags=re.IGNORECASE)

    # Drop vendor/platform repetition at the beginning
    s = re.sub(r"^\s*(insta(?:gram)?|instagram|tiktok|tik\s*tok|youtube|yt|twitter|x|facebook|telegram)\b\s*[-–—|:]?\s*", "", s, flags=re.IGNORECASE)
    # If there is still a platform prefix like 'Instagram - Likes', keep the right side
    s = re.sub(r"^\s*(instagram|tiktok|youtube|twitter|facebook|telegram)\s*[-–—]+\s*", "", s, flags=re.IGNORECASE)

    # Collapse whitespace and trim separators
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip(" -–—|:;,")
    return s.strip()

def rows2(buttons: List[InlineKeyboardButton]) -> List[List[InlineKeyboardButton]]:
    return [buttons[i:i + 2] for i in range(0, len(buttons), 2)]


def footer() -> List[List[InlineKeyboardButton]]:
    return [[
        InlineKeyboardButton("🔙 Back", callback_data="nav:back"),
        InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main"),
    ]]


async def _maintenance_blocker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Return True if request is blocked due to maintenance mode (non-admin)."""
    if is_admin(update, context):
        return False

    Session = context.application.bot_data.get("Session")
    if not Session:
        return False

    try:
        async with Session() as session:
            if not await is_maintenance_on(session):
                return False
    except Exception:
        # If we can't read settings, do not block users.
        return False

    msg = "⚙️ Maintenance Mode. We'll be back soon!"
    try:
        if update.callback_query:
            try:
                await update.callback_query.answer()
            except Exception:
                pass
            try:
                await context.application.bot.send_message(chat_id=update.effective_chat.id, text=msg)
            except Exception:
                pass
        elif update.message:
            await update.message.reply_text(msg)
        else:
            await context.application.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except Exception:
        pass
    return True


def footer_back_to_platforms() -> List[List[InlineKeyboardButton]]:
    # Explicit fix for the "platform menu back loop" bug:
    # Always go back to platform selection screen.
    return [[InlineKeyboardButton("🔙 Back", callback_data="nav:platforms")]]


def footer_admin() -> List[List[InlineKeyboardButton]]:
    # Explicit fix: admin submenus must go back to Admin root, not Main Menu.
    return [[
        InlineKeyboardButton("🔙 Back", callback_data="nav:admin_root"),
        InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main"),
    ]]


def nav_set(context: ContextTypes.DEFAULT_TYPE, screen: str, payload: Optional[Dict[str, Any]] = None, push: bool = True) -> None:
    """
    Basic stack navigation for generic "Back".

    We also de-duplicate pushes to prevent back-loops when a screen re-renders itself.
    """
    stack: List[Dict[str, Any]] = context.user_data.setdefault("nav_stack", [])
    cur = context.user_data.get("nav_current")
    nxt = {"screen": screen, "payload": payload or {}}

    if push and cur:
        if cur.get("screen") != nxt["screen"] or (cur.get("payload") or {}) != nxt["payload"]:
            stack.append(cur)

    context.user_data["nav_current"] = nxt


def nav_pop(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    stack: List[Dict[str, Any]] = context.user_data.setdefault("nav_stack", [])
    if stack:
        prev = stack.pop()
        context.user_data["nav_current"] = prev
        return prev
    context.user_data["nav_current"] = {"screen": MAIN, "payload": {}}
    return context.user_data["nav_current"]


async def send_or_edit(update: Update, text: str, kb: InlineKeyboardMarkup, parse_mode: str | None = None) -> None:
    """Edit callback message when possible; otherwise send a new message.

    Note: we keep this helper very small; for screens that must be Markdown,
    call edit_message_text/reply_text directly with parse_mode.
    """
    kwargs = {"reply_markup": kb, "disable_web_page_preview": True}
    if parse_mode:
        kwargs["parse_mode"] = parse_mode

    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(text, **kwargs)
            return
        except Exception:
            pass
    if update.effective_message:
        await update.effective_message.reply_text(text, **kwargs)


# -----------------------------
# Vendor display short-codes
# -----------------------------
VENDOR_SHORT = {
    "MoreThanPanel": "MTP",
    "Growfollows": "GW",
    "Smmpak": "SMP",
    "JustAnotherPanel": "JAP",
}


def vendor_tag(vendor_name: str) -> str:
    code = VENDOR_SHORT.get((vendor_name or "").strip(), None)
    if code:
        return f"[ {code} ]"
    # Fallback: first 6 chars of vendor name
    v = (vendor_name or "").strip() or "V"
    return f"[ {v[:6].upper()} ]"


async def send_main_menu_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a fresh Main Menu message (does not edit the previous message)."""
    Session = context.application.bot_data["Session"]
    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

    # Ensure row2_right is always defined

    row2_right = InlineKeyboardButton("⚙️ Support", callback_data="nav:support")


    if is_admin(update, context):
        row2_right = InlineKeyboardButton("⚙️ Admin", callback_data="nav:admin_root")
    else:
        row2_right = InlineKeyboardButton("⚙️ Support", callback_data="nav:support")

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🚀 Social Media Services", callback_data="nav:platforms"),
             InlineKeyboardButton("💼 Wallet", callback_data="nav:wallet")],
            [InlineKeyboardButton("📦 My Orders", callback_data="nav:orders"),
             row2_right],
        ]
    )
    bal = float(user.balance or 0)
    header = f"""⭐️ Welcome to your Social Media Growth Bot ⭐️
Your trusted marketplace for premium social engagement

<b>🪪 User ID:</b> {user.telegram_id}
<b>💰 Balance:</b> ${bal:.2f}"""
    if update.effective_chat:
        await update.effective_chat.send_message(header, reply_markup=kb, disable_web_page_preview=True, parse_mode="HTML")
def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    settings = context.application.bot_data["settings"]
    return bool(update.effective_user) and (update.effective_user.id in settings.admin_ids)


# -----------------------------
# Platform slug encoding (safe callback_data)
# -----------------------------
PLATFORM_SLUG = {
    "Instagram": "ig",
    "TikTok": "tt",
    "YouTube": "yt",
    "X(Twitter)": "x",
    "Facebook": "fb",
    "Telegram": "tg",
    "Spotify": "sp",
}
SLUG_PLATFORM = {v: k for k, v in PLATFORM_SLUG.items()}


# -----------------------------
# Token maps for category/subtype selection
# (keeps callback_data short + safe)
# -----------------------------
def _set_token_map(context: ContextTypes.DEFAULT_TYPE, key: str, values: List[str]) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for i, v in enumerate(values):
        mp[f"{key}{i}"] = v
    context.user_data[f"{key}_map"] = mp
    return mp


def _get_token(context: ContextTypes.DEFAULT_TYPE, key: str, token: str) -> Optional[str]:
    mp: Dict[str, str] = context.user_data.get(f"{key}_map") or {}
    return mp.get(token)


def _ui_subtype(db_value: str) -> str:
    s = (db_value or "").strip()
    return s if s else "Standard"


# -----------------------------
# Screens
# -----------------------------
async def screen_main(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, MAIN, {}, push=False)

    Session = context.application.bot_data["Session"]
    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

    # Admin button visibility rule:
    # Only show ⚙️ Admin to telegram_ids listed in ADMIN_IDS.
    if is_admin(update, context):
        row2_right = InlineKeyboardButton("⚙️ Admin", callback_data="nav:admin_root")
    else:
        # Keep 2-per-row layout for normal users
        row2_right = InlineKeyboardButton("⚙️ Support", callback_data="nav:support")

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🚀 Social Media Services", callback_data="nav:platforms"),
             InlineKeyboardButton("💼 Wallet", callback_data="nav:wallet")],
            [InlineKeyboardButton("📦 My Orders", callback_data="nav:orders"),
             row2_right],
        ]
    )
    bal = float(user.balance or 0)
    header = f"""⭐️ Welcome to your Social Media Growth Bot ⭐️
Your trusted marketplace for premium social engagement

<b>🪪 User ID:</b> {user.telegram_id}
<b>💰 Balance:</b> ${bal:.2f}"""
    await send_or_edit(update, header, kb, parse_mode="HTML")

async def screen_support(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, SUPPORT, {}, push=True)
    kb = InlineKeyboardMarkup(footer())
    await send_or_edit(
        update,
        "Welcome to your Social Media Enhancement\n\n"
        "Boost your stats on Telegram, Instagram, YouTube, and more with premium engagement. ✨\n\n"
        "Exclusive Features:\n"
        "✅ Instant Service: Orders delivered automatically.\n"
        "✅ Crypto Payments: Secure wallet & balance management.\n"
        "✅ Full Transparency: Track orders & history anytime.\n"
        "✅ 24/7 Uptime: Always active, always ready.\n"
        "📩 Need Help? DM @bandfullness for assistance.",
        kb,
    )


async def screen_platforms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, PLATFORMS, {}, push=True)
    btns: List[InlineKeyboardButton] = []
    for p in PLATFORM_LIST:
        slug = PLATFORM_SLUG.get(p, p.lower()[:10])
        btns.append(InlineKeyboardButton(p, callback_data=f"plat:{slug}"))

    kb = InlineKeyboardMarkup(rows2(btns) + footer())
    await send_or_edit(update, "🌐 Choose a Social Platform", kb)


async def screen_platform_categories(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str) -> None:
    """
    Dynamic categories per platform:
    Query DB and generate buttons based on existing services.
    """
    Session = context.application.bot_data["Session"]
    nav_set(context, PLAT_CATEGORIES, {"platform": platform}, push=True)

    async with Session() as session:
        rows = (await session.execute(
            select(Service.category)
            .where(Service.platform == platform, Service.is_active == True)
            .distinct()
            .order_by(Service.category.asc())
        )).all()
        categories = [r[0] for r in rows if r and r[0]]

        # Menu Manager integration: hide categories marked as not visible.
        vis_map: dict[str, bool] = {}
        try:
            mv_rows = (await session.execute(
                select(MenuVisibility.category, MenuVisibility.is_visible)
                .where(MenuVisibility.platform == platform, MenuVisibility.category.in_(categories))
            )).all()
            vis_map = {r[0]: bool(r[1]) for r in mv_rows if r and r[0]}
        except Exception:
            vis_map = {}

        if categories:
            categories = [c for c in categories if (vis_map.get(c, True) is True)]

    context.user_data["sel_platform"] = platform
    context.user_data.pop("sel_category", None)
    context.user_data.pop("sel_subtype", None)

    if not categories:
        kb = InlineKeyboardMarkup(footer_back_to_platforms() + [[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
        await send_or_edit(update, f"{platform}\n\nNo services found in DB for this platform.\n\nTry Admin → Catalog Sync.", kb)
        return

    # token map for callback safety
    cat_map = _set_token_map(context, "cat", categories)

    slug = PLATFORM_SLUG.get(platform, "ig")
    btns = [InlineKeyboardButton(_short(cat, 30), callback_data=f"cat:{slug}:{tok}") for tok, cat in cat_map.items()]
    kb = InlineKeyboardMarkup(rows2(btns) + footer_back_to_platforms() + [[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
    await send_or_edit(update, f"• {platform}\n\n🛍 Choose a Service", kb)


async def screen_platform_subtypes(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, category: str) -> None:
    """
    Sub-type step (always shown):
    Query DB for distinct sub_type under platform+category and show options.
    """
    Session = context.application.bot_data["Session"]
    nav_set(context, PLAT_SUBTYPES, {"platform": platform, "category": category}, push=True)

    async with Session() as session:
        rows = (await session.execute(
            select(Service.sub_type)
            .where(Service.platform == platform, Service.category == category, Service.is_active == True)
            .distinct()
            .order_by(Service.sub_type.asc())
        )).all()
        subtypes_raw = [r[0] for r in rows if r is not None]

    # normalize / ensure at least one subtype option
    subtypes: List[str] = []
    seen = set()
    for s in subtypes_raw:
        s2 = (s or "").strip()
        if s2 not in seen:
            seen.add(s2)
            subtypes.append(s2)
    if not subtypes:
        subtypes = [""]  # Standard

    context.user_data["sel_platform"] = platform
    context.user_data["sel_category"] = category
    context.user_data.pop("sel_subtype", None)

    sub_map = _set_token_map(context, "sub", subtypes)
    slug = PLATFORM_SLUG.get(platform, "ig")

    btns = [InlineKeyboardButton(_ui_subtype(st), callback_data=f"sub:{slug}:{tok}") for tok, st in sub_map.items()]

    # Back here should go to categories for THIS platform, not to platform list.
    kb = InlineKeyboardMarkup(
        rows2(btns)
        + [[InlineKeyboardButton("🔙 Back", callback_data=f"plat:{slug}")]]
        + [[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]]
    )
    await send_or_edit(update, f"• {platform} — {category}\n\n🏷️ Choose a Variant", kb)


async def screen_service_list(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, category: str, subtype: str) -> None:
    """
    Intermediate step:
    After selecting sub-type, show ALL matching specific services as buttons with prices.

    Updates:
      - Display vendor short code before the service name: [ CODE ] ServiceName - $Price
      - 1-per-row layout
      - [ Prev ] / [ Next ] pagination
    """
    Session = context.application.bot_data["Session"]

    # Normalize selector strings to match the (stripped) button text.
    # This prevents "empty list" issues when the DB contains trailing/leading whitespace.
    platform = (platform or "").strip()
    category = (category or "").strip()
    subtype = (subtype or "").strip()

    nav_set(context, SERVICE_LIST, {"platform": platform, "category": category, "subtype": subtype}, push=True)

    # Global markup is stored in the DB; per-service overrides are in Service.custom_markup.
    # We'll fetch the global value once and then use service.custom_markup when present.
    global_markup = 10.0

    # page state
    page = int(context.user_data.get("svc_page", 0) or 0)
    page_size = 8
    if page < 0:
        page = 0

    async with Session() as session:
        # Pull the current global markup from DB (fallback to 10% if missing)
        try:
            global_markup = float(await get_markup_percent(session))
        except Exception:
            global_markup = 10.0

        # total
        total_stmt = (
            select(func.count())
            .select_from(Service)
            .where(
                Service.platform == platform,
                Service.category == category,
                func.coalesce(func.trim(Service.sub_type), "") == subtype.strip(),
                Service.is_active == True,
            )
        )
        total = int((await session.execute(total_stmt)).scalar() or 0)

        stmt = (
            select(Service)
            .where(
                Service.platform == platform,
                Service.category == category,
                func.coalesce(func.trim(Service.sub_type), "") == subtype.strip(),
                Service.is_active == True,
            )
            .order_by(Service.rate.asc(), Service.id.asc())
            .offset(page * page_size)
            .limit(page_size)
        )
        services = list((await session.execute(stmt)).scalars().all())

    context.user_data["sel_platform"] = platform
    context.user_data["sel_category"] = category
    context.user_data["sel_subtype"] = subtype
    context.user_data["svc_page"] = page

    if total == 0 or not services:
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔙 Back", callback_data="nav:subtypes")],
             [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]]
        )
        await send_or_edit(update, f"{platform} — {category} — {_ui_subtype(subtype)}\n\nNo services found for this selection.\n\nTry Admin → Catalog Sync.", kb)
        return

    svc_rows: List[List[InlineKeyboardButton]] = []
    # show per-row service with FULL website name + price per 1k
    # IMPORTANT: do not abbreviate/shorten the name (prevents confusing UX)
    skipped = 0
    async with async_session() as ses2:
        for s in services:
            # Some variants (often "Refill"/"30d") may include services that are missing
            # an active vendor mapping. If we let pick_cheapest_vendor raise here, the
            # callback handler exits and the UI appears "stuck".
            try:
                vs, v = await pick_cheapest_vendor(ses2, s.id)
            except Exception as e:
                skipped += 1
                try:
                    logging.warning("Skipping service id=%s in list due to vendor mapping/pricing error: %s", getattr(s, "id", "?"), e)
                except Exception:
                    pass
                continue

            try:
                effective_markup = float(s.custom_markup) if s.custom_markup is not None else float(global_markup)
                rate_per_1000 = apply_markup(float(vs.vendor_rate), effective_markup)
            except Exception as e:
                skipped += 1
                try:
                    logging.warning("Skipping service id=%s in list due to rate/markup error: %s", getattr(s, "id", "?"), e)
                except Exception:
                    pass
                continue

            full_name = (s.name or "").strip()
            if not full_name:
                # fallback (should rarely happen)
                full_name = f"{(s.platform or '').strip()} [{(s.sub_type or 'Standard').strip() or 'Standard'}]".strip()

            # Smart truncation so the price is always visible at the end of the button.
            # Target ~40-50 chars total.
            price_part = f" - ${float(rate_per_1000):.2f}/k"
            max_total = 50
            max_name_len = max(10, max_total - len(price_part))
            name_part = full_name
            if len(name_part) > max_name_len:
                cut = max(0, max_name_len - 3)
                name_part = (name_part[:cut].rstrip() + "...") if cut > 0 else "..."

            label = f"{name_part}{price_part}".strip()
            svc_rows.append([InlineKeyboardButton(label, callback_data=f"svc:{s.id}")])

    # If everything on this page was skipped (e.g., missing vendor mappings), don't leave the UI idle.
    if not svc_rows:
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔙 Back", callback_data="nav:subtypes")],
             [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]]
        )
        await send_or_edit(
            update,
            f"• {platform} — {category} — {_ui_subtype(subtype)}\n\nNo available packages for this variant right now.\n\nTry another variant or run Admin → Catalog Sync.",
            kb,
        )
        return

    kb_rows: List[List[InlineKeyboardButton]] = list(svc_rows)

    nav_row: List[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("[ Prev ]", callback_data=f"svcpage:{page-1}"))
    if (page + 1) * page_size < total:
        nav_row.append(InlineKeyboardButton("[ Next ]", callback_data=f"svcpage:{page+1}"))
    if nav_row:
        kb_rows.append(nav_row)

    kb_rows += [[InlineKeyboardButton("🔙 Back", callback_data="nav:subtypes")]]
    kb_rows += [[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]]

    kb = InlineKeyboardMarkup(kb_rows)
    start_i = page * page_size + 1
    end_i = min(total, (page + 1) * page_size)
    # 2-line caption format (breadcrumb then instruction + pagination range)
    await send_or_edit(
        update,
        f"• {platform} — {category} — {_ui_subtype(subtype)}\n\n📦 Select a Package ({start_i}-{end_i}/{total})",
        kb,
    )

async def screen_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    Session = context.application.bot_data["Session"]
    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

    nav_set(context, WALLET, {}, push=True)
    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💵 Deposit", callback_data="nav:deposit"),
             InlineKeyboardButton("📜 Transactions", callback_data="nav:tx")],
        ] + footer()
    )
    await send_or_edit(update, f"💼 Wallet\n\nBalance: {money(float(user.balance))}", kb)


async def screen_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, DEPOSIT, {}, push=True)

    coins = ["BTC", "USDT", "ETH", "TRX", "TON", "LTC"]
    btns = [InlineKeyboardButton(c, callback_data=f"dep:coin:{c}") for c in coins]
    kb = InlineKeyboardMarkup(rows2(btns) + footer())
    await send_or_edit(update, "Choose a coin to deposit with:", kb)



async def screen_deposit_amounts(update: Update, context: ContextTypes.DEFAULT_TYPE, coin: str) -> None:
    coin = (coin or "").upper()
    min_amount = 5 if coin == "TRX" else 1
    presets = [1, 2, 5, 10, 20, 50, 100]
    presets = [p for p in presets if p >= min_amount]
    btns = [InlineKeyboardButton(f"${p}", callback_data=f"dep:amt:{coin}:{p}") for p in presets]
    btns.append(InlineKeyboardButton("Custom", callback_data=f"dep:custom:{coin}"))
    kb = InlineKeyboardMarkup(rows2(btns) + footer())
    await send_or_edit(update, f"Choose deposit amount in USD (min ${min_amount}) for {coin}:", kb)

async def screen_transactions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    Session = context.application.bot_data["Session"]
    nav_set(context, TX, {}, push=True)

    page = int(context.user_data.get("tx_page", 0) or 0)
    page_size = 5
    if page < 0:
        page = 0

    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

        total_stmt = select(func.count()).select_from(Transaction).where(Transaction.user_id == user.id)
        total = int((await session.execute(total_stmt)).scalar() or 0)

        stmt = (
            select(Transaction)
            .where(Transaction.user_id == user.id)
            .order_by(desc(Transaction.created_at))
            .offset(page * page_size)
            .limit(page_size)
        )
        txs = list((await session.execute(stmt)).scalars().all())

    context.user_data["tx_page"] = page

    lines: List[str] = ["Transactions", ""]
    if total == 0 or not txs:
        lines.append("No transactions.")
    else:
        for t in txs:
            try:
                p = json.loads(t.payment_payload or "{}")
                inv = p.get("invoice_id", "?")
                asset = (p.get("asset", "") or "").upper()
                pay_url = p.get("pay_url", "") or ""
            except Exception:
                inv, asset, pay_url = "?", "", ""

            ts = t.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            st = str(t.status or "").lower().strip()
            is_paid = st in ("completed", "paid", "success")
            is_failed = st in ("failed", "error", "expired", "canceled", "cancelled")
            is_processing = st in ("processing", "confirming")

            if is_paid:
                status_txt = "✅ Paid"
            elif is_failed:
                status_txt = "❌ Failed"
            elif is_processing:
                status_txt = "⏳ Processing"
            else:
                status_txt = "⏳ Unpaid"

            amount_txt = money(float(t.amount))
            safe_ts = _html.escape(ts)
            safe_asset = _html.escape(asset)
            safe_inv = _html.escape(str(inv))
            safe_status = _html.escape(status_txt)

            line_txt = f"- {safe_ts} | {safe_status} | {amount_txt} {safe_asset} | invoice {safe_inv}"

            is_unpaid = (not is_paid) and (not is_failed) and (not is_processing)
            if is_unpaid and pay_url:
                safe_url = _html.escape(pay_url, quote=True)
                line_txt += f' | <a href="{safe_url}">Pay Now</a>'

            lines.append(line_txt)

    kb_rows: List[List[InlineKeyboardButton]] = []
    nav_row: List[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("[ Prev ]", callback_data=f"txpage:{page-1}"))
    if (page + 1) * page_size < total:
        nav_row.append(InlineKeyboardButton("[ Next ]", callback_data=f"txpage:{page+1}"))
    if nav_row:
        kb_rows.append(nav_row)
    kb_rows += footer()
    kb = InlineKeyboardMarkup(kb_rows)

    text_out = "\n".join(lines)
    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                text_out,
                reply_markup=kb,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            return
        except Exception:
            pass
    if update.effective_message:
        await update.effective_message.reply_text(
            text_out,
            reply_markup=kb,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )

async def screen_orders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    Session = context.application.bot_data["Session"]
    nav_set(context, ORDERS, {}, push=True)

    page = int(context.user_data.get("orders_page", 0) or 0)
    page_size = 4
    if page < 0:
        page = 0

    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

        total_stmt = select(func.count()).select_from(Order).where(Order.user_id == user.id)
        total = int((await session.execute(total_stmt)).scalar() or 0)

        stmt = (
            select(Order)
            .where(Order.user_id == user.id)
            .order_by(desc(Order.created_at))
            .offset(page * page_size)
            .limit(page_size)
        )
        orders = list((await session.execute(stmt)).scalars().all())

    context.user_data["orders_page"] = page

    lines = ["My Orders", ""]
    if total == 0 or not orders:
        lines.append("No orders.")
    else:
        for idx, o in enumerate(orders, start=1):
            ts = o.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            st = (o.status or "").strip()
            st_l = st.lower()
            if st_l in ("completed", "complete", "done", "success", "succeeded"):
                emo = "✅"
            elif st_l in ("failed", "fail", "canceled", "cancelled", "refunded"):
                emo = "❌"
            elif st_l in ("processing", "pending", "in progress", "inprogress"):
                emo = "⏳"
            else:
                emo = ""
            st_disp = f"{emo} {st}".strip()
            # Newest first (created_at DESC), numbered 1..N on the page.
            lines.append(f"{idx}. {o.id} | {st_disp} | {money(float(o.charge))} | {ts}")
    # Per requirement: pagination controls ONLY (no numbered/order-id buttons).
    kb_rows: List[List[InlineKeyboardButton]] = []

    nav_row: List[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️ Back", callback_data=f"ordpage:{page-1}"))
    if (page + 1) * page_size < total:
        nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"ordpage:{page+1}"))
    if nav_row:
        kb_rows.append(nav_row)

    kb_rows.append([InlineKeyboardButton("🔍 Check Order", callback_data="nav:check_order")])
    kb_rows += footer()
    kb = InlineKeyboardMarkup(kb_rows)
    await send_or_edit(update, "\n".join(lines), kb)


async def screen_order_detail(update: Update, context: ContextTypes.DEFAULT_TYPE, order_id: int) -> None:
    Session = context.application.bot_data["Session"]
    nav_set(context, ORDER_DETAIL, {"order_id": order_id}, push=True)

    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)
        o = (await session.execute(select(Order).where(Order.id == order_id, Order.user_id == user.id))).scalars().first()

    if not o:
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Order not found.", kb)
        return

    text_out = (
        f"📦 Order #{o.id}\n\n"
        f"Status: {o.status}\n"
        f"Charge: {money(float(o.charge))}\n"
        f"Cost: {money(float(o.cost))}\n"
        f"Profit: {money(float(o.profit))}\n"
        f"Quantity: {o.quantity}\n"
        f"Link: {o.link}\n\n"
        f"Vendor: {o.vendor_name or '-'}\n"
        f"Vendor Order ID: {o.vendor_order_id or '-'}"
    )
    kb = InlineKeyboardMarkup(footer())
    await send_or_edit(update, text_out, kb)


# -----------------------------
# Admin screens
# -----------------------------
async def screen_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    nav_set(context, ADMIN, {}, push=True)

    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    # Maintenance status (for button label)
    Session = context.application.bot_data["Session"]
    maint_on = False
    try:
        async with Session() as session:
            maint_on = await is_maintenance_on(session)
    except Exception:
        maint_on = False

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Markup %", callback_data="adm:markup"),
             InlineKeyboardButton("Vendors", callback_data="adm:vendors")],
            [InlineKeyboardButton("Catalog Sync", callback_data="adm:sync"),
             InlineKeyboardButton("Add Funds", callback_data="adm:addfunds")],
            [InlineKeyboardButton("Set Service Markup", callback_data="adm:svcmarkup")],
            [InlineKeyboardButton("🔧 Menu Manager", callback_data="adm:menumgr")],
            [InlineKeyboardButton(f"⚙️ Maintenance Mode ({'ON' if maint_on else 'OFF'})", callback_data="adm:maint")],
            [InlineKeyboardButton("Daily Stats", callback_data="adm:dailystats"),
             InlineKeyboardButton("Check Stats", callback_data="adm:vendorstats")],
        ] + footer()
    )
    await send_or_edit(update, "⚙️ Admin Panel", kb)

async def screen_admin_vendors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, ADMIN_VENDORS, {}, push=True)

    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Add Vendor", callback_data="adm:addvendor"),
             InlineKeyboardButton("Test Vendor", callback_data="adm:testvendor")],
            [InlineKeyboardButton("Toggle Status", callback_data="adm:togglevendor"),
             InlineKeyboardButton("Check Stats", callback_data="adm:vendorstats")],
        ] + footer_admin()
    )
    await send_or_edit(update, "Vendors", kb)

async def screen_menu_manager(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: Toggle visibility of (platform, category) buttons shown to users."""
    if await _maintenance_blocker(update, context):
        return

    nav_set(context, ADMIN_MENU_MANAGER, {}, push=True)

    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    Session = context.application.bot_data["Session"]
    page_size = 20
    page = int(context.user_data.get("mm_page") or 0)

    async with Session() as session:
        combos = (await session.execute(
            select(Service.platform, Service.category)
            .where(Service.platform.is_not(None), Service.category.is_not(None))
            .distinct()
            .order_by(Service.platform.asc(), Service.category.asc())
        )).all()
        combos = [(p, c) for (p, c) in combos if p and c]

        ac_rows = (await session.execute(
            select(Service.platform, Service.category, func.count(Service.id))
            .where(Service.is_active == True)
            .group_by(Service.platform, Service.category)
        )).all()
        active_counts = {(p, c): int(n or 0) for (p, c, n) in ac_rows}

        try:
            mv_rows = (await session.execute(
                select(MenuVisibility.platform, MenuVisibility.category, MenuVisibility.is_visible)
            )).all()
        except Exception:
            logging.exception("Menu Manager: failed to load MenuVisibility")
            kb = InlineKeyboardMarkup(footer())
            await send_or_edit(update, "⚠️ Error loading menu data. Please contact support.", kb)
            return
        mv_map = {(p, c): bool(v) for (p, c, v) in mv_rows if p and c}


    total = len(combos)
    max_page = max(0, (total - 1) // page_size) if total else 0
    if page < 0:
        page = 0
    if page > max_page:
        page = max_page
    context.user_data["mm_page"] = page

    start = page * page_size
    end = start + page_size
    slice_combos = combos[start:end]
    context.user_data["mm_combos"] = slice_combos

    lines = ["📦 Menu Categories", ""]
    if not combos:
        lines.append("No categories found in services table.")
    else:
        for (plat, cat) in slice_combos:
            n = active_counts.get((plat, cat), 0)
            visible = mv_map.get((plat, cat), True if n > 0 else False)
            vis_txt = "🟢 Visible" if visible else "🔴 Hidden"
            lines.append(f"{_plat_display(plat)} - {cat} [Active: {n}] [{vis_txt}]")

    btn_rows: list[list[InlineKeyboardButton]] = []
    for i, (plat, cat) in enumerate(slice_combos):
        btn_rows.append([
            InlineKeyboardButton(f"{_plat_display(plat)} - {cat}", callback_data=f"mm:nop:{i}"),
            InlineKeyboardButton("Toggle", callback_data=f"mm:t:{i}"),
        ])

    nav_row: list[InlineKeyboardButton] = []
    if combos and max_page > 0:
        if page > 0:
            nav_row.append(InlineKeyboardButton("⬅️ Prev", callback_data="mm:page:prev"))
        if page < max_page:
            nav_row.append(InlineKeyboardButton("Next ➡️", callback_data="mm:page:next"))
    if nav_row:
        btn_rows.append(nav_row)

    kb = InlineKeyboardMarkup(btn_rows + footer_admin())
    await send_or_edit(update, "\n".join(lines), kb)



async def screen_vendor_list(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    Session = context.application.bot_data["Session"]

    async with Session() as session:
        vendors = await list_all_vendors(session)

    nav_set(context, ADMIN_VENDOR_PICK, {"mode": mode}, push=True)
    context.user_data["vendor_pick_mode"] = mode

    if not vendors:
        kb = InlineKeyboardMarkup(footer_admin())
        await send_or_edit(update, "No vendors yet.", kb)
        return

    btns = []
    for v in vendors:
        status = "ON" if v.is_active else "OFF"
        btns.append(InlineKeyboardButton(f"{v.name} [{status}]", callback_data=f"vend:{v.id}"))

    kb = InlineKeyboardMarkup(rows2(btns) + footer_admin())
    await send_or_edit(update, "Select a vendor:", kb)


async def screen_vendor_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, ADMIN_VENDOR_STATS, {}, push=True)

    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    Session = context.application.bot_data["Session"]
    async with Session() as session:
        vendors = await list_all_vendors(session)

    lines = ["📊 Vendor Status", ""]
    if not vendors:
        lines.append("No vendors.")
    else:
        for v in vendors:
            status = "ON" if v.is_active else "OFF"
            test = "—"
            if v.last_test_ok is True:
                test = "OK"
            elif v.last_test_ok is False:
                test = "FAIL"
            sync = "—"
            if v.last_sync_ok is True:
                sync = "OK"
            elif v.last_sync_ok is False:
                sync = "FAIL"
            err = (v.last_error or "")
            if len(err) > 80:
                err = err[:80] + "…"
            lines.append(f"- {v.name} [{status}] | test:{test} | sync:{sync} | svcs:{v.last_services_count or 0} | {err}")

    kb = InlineKeyboardMarkup(footer_admin())
    await send_or_edit(update, "\n".join(lines), kb)


async def screen_daily_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nav_set(context, ADMIN_DAILY_STATS, {}, push=True)

    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    Session = context.application.bot_data["Session"]

    # Date navigation (UTC)
    # Stored as YYYY-MM-DD string in user_data.
    sel = str(context.user_data.get("admin_daily_stats_date") or "")
    if sel:
        try:
            y, m, d = [int(x) for x in sel.split("-")]
            start = datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            sel = ""

    if not sel:
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        context.user_data["admin_daily_stats_date"] = start.strftime("%Y-%m-%d")

    end = start + timedelta(days=1)

    async with Session() as session:
        total_orders = (await session.execute(
            select(func.count()).select_from(Order).where(Order.created_at >= start, Order.created_at < end)
        )).scalar() or 0
        total_profit = (await session.execute(
            select(func.coalesce(func.sum(Order.profit), 0.0)).where(Order.created_at >= start, Order.created_at < end)
        )).scalar() or 0.0
        total_deposits = (await session.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0.0))
            .where(Transaction.status == "completed", Transaction.created_at >= start, Transaction.created_at < end)
        )).scalar() or 0.0

    # Prev/Next day navigation
    nav_row = [
        InlineKeyboardButton("🔙 Prev Day", callback_data="adm:dailystats:prev"),
        InlineKeyboardButton("🔜 Next Day", callback_data="adm:dailystats:next"),
    ]
    kb = InlineKeyboardMarkup([nav_row] + footer_admin())
    await send_or_edit(
        update,
        f"📈 Daily Stats (UTC)\n\nDate: {start.strftime('%Y-%m-%d')}\nTotal orders: {int(total_orders)}\nTotal deposits: {money(float(total_deposits))}\nTotal profit: {money(float(total_profit))}",
        kb,
    )


# -----------------------------
# Payments (Crypto Pay / Send.tg)
# -----------------------------
async def _deposit_invoice_task(
    app: Application,
    chat_id: int,
    message_id: int,
    user_id: int,
    username: str | None,
    amount_usd: float,
    coin: str,
) -> None:
    """Create a Crypto Pay invoice (priced in USD) and persist a pending Transaction.

    Runs in background via asyncio.create_task() so the bot stays responsive.
    """
    settings = app.bot_data["settings"]
    Session = app.bot_data["Session"]

    asset = (coin or "USDT").strip().upper()
    crypto = CryptoPayClient(api_token=settings.cryptopay_token)

    try:
        # IMPORTANT: The user selects the amount in USD.
        # Create a FIAT invoice so Crypto Pay converts USD -> selected asset automatically.
        payload_str = f"tg:{user_id}:{uuid.uuid4()}"
        payload = {
            "currency_type": "fiat",
            "fiat": "USD",
            "amount": str(float(amount_usd)),
            "accepted_assets": asset,
            "description": f"Deposit ${float(amount_usd):.2f} via {asset}",
            "payload": payload_str,
            "allow_comments": False,
            "allow_anonymous": False,
        }
        inv = await crypto._call("createInvoice", payload)
        invoice_id = str(inv["invoice_id"])
        pay_url = inv.get("pay_url") or inv.get("bot_invoice_url") or ""
        # Show the approved deposit invoice message (HTML)

        amount = float(amount_usd)

        currency = asset
        text_html = (
            "💳 Deposit Invoice Created\n\n"
            f"Amount (USD): ${amount:.2f}\n"
            f"Currency: {currency}\n\n"
            "• Complete the payment below to finalize your deposit.\n"
            "• Your balance updates instantly upon confirmation.\n\n"
            '<a href="https://telegra.ph/How-to-Deposit-02-10">View Payment Guide</a>'
        )

        pay_rows = []
        if pay_url:
            pay_rows.append([InlineKeyboardButton("💳 Pay Now", url=pay_url)])
        pay_rows.append([InlineKeyboardButton("⬅️ Back", callback_data="nav:wallet")])
        pay_kb = InlineKeyboardMarkup(pay_rows)

        # Persist pending transaction (invoice is unpaid at creation)
        async with Session() as session:
            user = await get_or_create_user(session, user_id, username)
            await create_deposit_transaction(
                session,
                user,
                float(amount_usd),
                invoice_id,
                asset,
                pay_url,
                payload=payload_str,
            )

        # Send a NEW message for the invoice details (do not edit existing messages).
        await app.bot.send_message(
            chat_id=chat_id,
            text=text_html,
            reply_markup=pay_kb,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )

        # Delete the temporary 'Creating invoice...' message to keep chat clean.
        try:
            await app.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            pass

    except Exception as e:
        # Always surface the error to the admin/user instead of failing silently.
        try:
            await app.bot.send_message(chat_id=chat_id, text=f"❌ Failed to create invoice: {e}")
        except Exception:
            pass

async def create_invoice(update: Update, context: ContextTypes.DEFAULT_TYPE, amount_usd: float, coin: str) -> None:
    """Start invoice creation in the background and respond immediately."""
    q = update.callback_query
    chat_id = int(update.effective_chat.id)
    message_id = int(q.message.message_id) if q and q.message else int(update.effective_message.message_id)
    user_id = int(update.effective_user.id)
    username = update.effective_user.username

    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("💼 Wallet", callback_data="nav:wallet")]] + footer()
    )
    await send_or_edit(update, "⏳ Creating invoice...", kb)

    try:
        context.application.create_task(
            _deposit_invoice_task(context.application, chat_id, message_id, user_id, username, float(amount_usd), str(coin))
        )
    except Exception:
        asyncio.create_task(
            _deposit_invoice_task(context.application, chat_id, message_id, user_id, username, float(amount_usd), str(coin))
        )

async def check_payment(update: Update, context: ContextTypes.DEFAULT_TYPE, invoice_id: int) -> None:
    settings = context.application.bot_data["settings"]
    Session = context.application.bot_data["Session"]
    crypto = CryptoPayClient(api_token=settings.cryptopay_token)

    try:
        inv = await crypto.get_invoice(invoice_id)
    except Exception as e:
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, f"Failed to check invoice: {e}", kb)
        return

    status = str(inv.get("status", "unknown"))

    async with Session() as session:
        tx = None
        txs = list((await session.execute(select(Transaction).order_by(desc(Transaction.created_at)).limit(200))).scalars().all())
        for t in txs:
            try:
                p = json.loads(t.payment_payload)
                if int(p.get("invoice_id", -1)) == invoice_id:
                    tx = t
                    break
            except Exception:
                continue

        if status == "paid" and tx and tx.status not in ("completed", "paid"):
            usr = (await session.execute(select(User).where(User.id == tx.user_id))).scalars().first()
            if usr:
                usr.balance = float(usr.balance) + float(tx.amount)
            tx.status = "completed"
            await commit_with_retry(session)

        if status == "expired" and tx and tx.status == "pending":
            tx.status = "expired"
            await commit_with_retry(session)

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Check Again", callback_data=f"pay:{invoice_id}"),
             InlineKeyboardButton("💼 Wallet", callback_data="nav:wallet")],
        ] + footer()
    )
    if status == "paid":
        await send_or_edit(update, "✅ Paid. Balance credited.", kb)
    else:
        await send_or_edit(update, f"Invoice {invoice_id} status: {status}", kb)


# -----------------------------
# Handlers / routers
# -----------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.delete()
    except Exception:
        pass

    if await _maintenance_blocker(update, context):
        return

    # Clean up the previous Main Menu message (to avoid clutter)
    try:
        prev_mid = context.user_data.get("last_main_menu_msg_id")
        if prev_mid and update.effective_chat:
            await context.application.bot.delete_message(chat_id=update.effective_chat.id, message_id=int(prev_mid))
    except Exception:
        pass

    Session = context.application.bot_data["Session"]

    # Send a fresh Main Menu message and remember it for next /start cleanup
    nav_set(context, MAIN, {}, push=False)
    async with Session() as session:
        user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

    if is_admin(update, context):
        row2_right = InlineKeyboardButton("⚙️ Admin", callback_data="nav:admin_root")
    else:
        row2_right = InlineKeyboardButton("⚙️ Support", callback_data="nav:support")

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🚀 Social Media Services", callback_data="nav:platforms"),
             InlineKeyboardButton("💼 Wallet", callback_data="nav:wallet")],
            [InlineKeyboardButton("📦 My Orders", callback_data="nav:orders"),
             row2_right],
        ]
    )
    bal = float(user.balance or 0)
    header = f"""⭐️ Welcome to your Social Media Growth Bot ⭐️
Your trusted marketplace for premium social engagement

<b>🪪 User ID:</b> {user.telegram_id}
<b>💰 Balance:</b> ${bal:.2f}"""
    if update.effective_chat:
        msg = await context.application.bot.send_message(
            chat_id=update.effective_chat.id,
            text=header,
            reply_markup=kb,
            disable_web_page_preview=True,
            parse_mode="HTML",
        )
        context.user_data["last_main_menu_msg_id"] = msg.message_id


async def nav_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    cmd = q.data

    if cmd == "nav:main":
        context.user_data["nav_stack"] = []
        context.user_data["input_mode"] = INP_NONE
        try:
            await update.callback_query.message.delete()
        except Exception:
            pass
        await screen_main(update, context)
        return

    if cmd == "nav:back":
        prev = nav_pop(context)
        await render_state(update, context, prev["screen"], prev.get("payload") or {})
        return

    if cmd == "nav:platforms":
        # Treat platform list as the root of "Social Media Services" to avoid Back-loop.
        context.user_data["nav_stack"] = []
        context.user_data["nav_current"] = {"screen": MAIN, "payload": {}}
        await screen_platforms(update, context)
        return

    if cmd == "nav:categories":
        plat = context.user_data.get("sel_platform")
        if plat:
            await screen_platform_categories(update, context, str(plat))
        else:
            await screen_platforms(update, context)
        return

    if cmd == "nav:subtypes":
        plat = context.user_data.get("sel_platform")
        cat = context.user_data.get("sel_category")
        if plat and cat is not None:
            await screen_platform_subtypes(update, context, str(plat), str(cat))
        else:
            await screen_platforms(update, context)
        return

    if cmd == "nav:services":
        plat = context.user_data.get("sel_platform")
        cat = context.user_data.get("sel_category")
        sub = context.user_data.get("sel_subtype", "")
        if plat and cat is not None:
            await screen_service_list(update, context, str(plat), str(cat), str(sub or ""))
        else:
            await screen_platforms(update, context)
        return

    if cmd == "nav:wallet":
        await screen_wallet(update, context)
        return

    if cmd == "nav:deposit":
        await screen_deposit(update, context)
        return

    if cmd == "nav:tx":
        await screen_transactions(update, context)
        return

    if cmd == "nav:orders":
        await screen_orders(update, context)
        return

    
    if cmd == "nav:check_order":
        context.user_data["input_mode"] = INP_CHECK_ORDER
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("❌ Cancel", callback_data="nav:main")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
            ]
        )
        await q.message.reply_text("Send your Order ID.", reply_markup=kb)
        return
    if cmd == "nav:support":
        await screen_support(update, context)
        return

    if cmd == "nav:admin_root":
        # Treat admin root as a top-level screen to avoid Back-loop from submenus.
        context.user_data["nav_stack"] = []
        context.user_data["nav_current"] = {"screen": MAIN, "payload": {}}
        await screen_admin(update, context)
        return


async def plat_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        return
    slug = parts[1]
    platform = SLUG_PLATFORM.get(slug)
    if not platform:
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Invalid platform.", kb)
        return
    await screen_platform_categories(update, context, platform)


async def cat_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    parts = data.split(":")
    if len(parts) < 3:
        return
    _, slug, token = parts[0], parts[1], parts[2]
    platform = SLUG_PLATFORM.get(slug)
    category = _get_token(context, "cat", token)
    if not platform or category is None:
        kb = InlineKeyboardMarkup(footer_back_to_platforms() + [[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
        await send_or_edit(update, "Selection expired. Please pick again.", kb)
        return
    context.user_data["sel_platform"] = platform
    context.user_data["sel_category"] = category
    await screen_platform_subtypes(update, context, platform, category)


async def sub_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    parts = data.split(":")
    if len(parts) < 3:
        return
    _, slug, token = parts[0], parts[1], parts[2]
    platform = SLUG_PLATFORM.get(slug)
    subtype = _get_token(context, "sub", token)
    category = context.user_data.get("sel_category")
    if not platform or subtype is None or category is None:
        kb = InlineKeyboardMarkup(footer_back_to_platforms() + [[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
        await send_or_edit(update, "Selection expired. Please pick again.", kb)
        return
    context.user_data["sel_platform"] = platform
    context.user_data["sel_subtype"] = subtype
    await screen_service_list(update, context, platform, str(category), str(subtype or ""))




async def svcpage_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    try:
        _, page_s = data.split(":", 1)
        page = int(page_s)
    except Exception:
        page = 0

    platform = context.user_data.get("sel_platform")
    category = context.user_data.get("sel_category")
    subtype = context.user_data.get("sel_subtype")
    if platform is None or category is None or subtype is None:
        await screen_platforms(update, context)
        return

    context.user_data["svc_page"] = page
    await screen_service_list(update, context, str(platform), str(category), str(subtype))


async def ordpage_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    try:
        _, page_s = data.split(":", 1)
        page = int(page_s)
    except Exception:
        page = 0
    context.user_data["orders_page"] = max(0, page)
    await screen_orders(update, context)


async def txpage_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    try:
        _, page_s = data.split(":", 1)
        page = int(page_s)
    except Exception:
        page = 0
    context.user_data["tx_page"] = max(0, page)
    await screen_transactions(update, context)

async def svc_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""

    try:
        service_id = int(data.split(":", 1)[1])
    except Exception:
        await send_or_edit(update, "Invalid service selection.", InlineKeyboardMarkup(footer()))
        return

    Session = context.application.bot_data["Session"]

    async with Session() as session:
        svc = (await session.execute(select(Service).where(Service.id == service_id))).scalars().first()
        if not svc:
            await send_or_edit(update, "Service not found.", InlineKeyboardMarkup(footer()))
            return
        try:
            vs, vendor = await pick_cheapest_vendor(session, svc.id)
        except RuntimeError:
            await update.effective_message.reply_text('❌ This service is temporarily unavailable . Please pick another service.')
            settings = context.application.bot_data.get('settings') or load_settings()
            alert_text = f"🚨 CRITICAL: No active vendor mapping for Service ID {svc.id}. Please fix immediately."
            for admin_id in (getattr(settings, 'admin_ids', None) or []):
                try:
                    await context.application.bot.send_message(chat_id=admin_id, text=alert_text)
                except Exception:
                    pass
            return ConversationHandler.END
        # Display pricing must respect per-service custom markup if set.
        try:
            global_markup = float(await get_markup_percent(session))
        except Exception:
            global_markup = float(context.application.bot_data.get("markup_percent", 10.0) or 10.0)
        effective_markup = float(svc.custom_markup) if svc.custom_markup is not None else float(global_markup)
        display_rate = apply_markup(float(vs.vendor_rate), effective_markup)

    context.user_data["order_service_id"] = service_id
    context.user_data["order_vendor_name"] = vendor.name
    context.user_data["order_price_display"] = float(display_rate)
    context.user_data["input_mode"] = INP_NONE

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✅ Confirm", callback_data="oc:confirm"),
             InlineKeyboardButton("❌ Cancel", callback_data="oc:cancel")],
            [InlineKeyboardButton("🔙 Back", callback_data="nav:services"),
             InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
        ]
    )
    # Details for confirmation (display-only)
    try:
        min_q = int(svc.min or 0)
    except Exception:
        min_q = 0
    try:
        max_q = int(svc.max or 0)
    except Exception:
        max_q = 0

    try:
        min_total = float(calc_charge(float(display_rate), int(min_q or 0))) if min_q else 0.0
    except Exception:
        min_total = 0.0

    name_l = f"{svc.name} {svc.sub_type or ''} {svc.type or ''}"
    refill_yes = "Yes" if re.search(r"\brefill\b", name_l, re.I) else "No"
    drip_yes = "Yes" if re.search(r"\bdrip", name_l, re.I) else "No"

    cat_line = f"{svc.category}"
    if (svc.sub_type or "").strip():
        cat_line += f" • {svc.sub_type}"

    svc_name = html.escape(str(svc.name or ""))
    svc_platform = html.escape(str(svc.platform or ""))
    cat_line_safe = html.escape(str(cat_line or ""))
    price_safe = html.escape(f"{money(float(display_rate))}/1k")
    min_total_safe = html.escape(money(float(min_total)))
    refill_yes_safe = html.escape(str(refill_yes))
    drip_yes_safe = html.escape(str(drip_yes))

    text_out = (
        "🧾 <b>Confirm Order</b>\n\n"
        f"📦 <b>Service:</b> {svc_name}\n"
        f"🏷️ <b>Platform:</b> {svc_platform}\n"
        f"📚 <b>Category:</b> {cat_line_safe}\n"
        f"💰 <b>Price:</b> {price_safe}\n"
        f"💳 <b>Min Total ({min_q}):</b> {min_total_safe}\n"
        f"📏 <b>Limits:</b> {min_q} - {max_q}\n"
        f"🧪 <b>Specs:</b> Refill: {refill_yes_safe} • Dripfeed: {drip_yes_safe}\n\n"
        "Tap <b>✅ Confirm</b> to continue and send your link/username."
    )
    await update.effective_chat.send_action(ChatAction.TYPING)
    await update.effective_chat.send_message(
        text_out,
        reply_markup=kb,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )



async def oc_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""

    if data == "oc:cancel":
        for k in ("order_service_id", "order_vendor_name", "order_price_display", "order_link", "order_qty"):
            context.user_data.pop(k, None)
        context.user_data["input_mode"] = INP_NONE

        platform = context.user_data.get("sel_platform")
        if platform:
            await screen_platform_categories(update, context, str(platform))
        else:
            await screen_platforms(update, context)
        return

    if data == "oc:confirm":
        service_id = int(context.user_data.get("order_service_id", 0) or 0)
        if not service_id:
            await send_or_edit(update, "No service selected.", InlineKeyboardMarkup(footer()))
            return

        context.user_data["input_mode"] = INP_ORDER_LINK
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("❌ Cancel", callback_data="oc:cancel")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
            ]
        )
        await update.effective_chat.send_message("Send your link/username:", reply_markup=kb)
        return

# -----------------------------
# Linear Order Conversation Flow
# Service -> Link -> Quantity -> Confirm/Cancel
# -----------------------------

def _is_valid_order_link(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False

    # Accept URLs
    try:
        u = urlparse(s)
        if u.scheme in ("http", "https") and bool(u.netloc):
            return True
    except Exception:
        pass

    # Accept usernames/handles (common for SMM: @user, user, user_name)
    if re.fullmatch(r"@?[A-Za-z0-9_\.]{3,64}", s):
        return True

    return False


async def order_conv_entry_from_service(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 1: Service Selection -> ask for Link."""
    if await _maintenance_blocker(update, context):
        return ConversationHandler.END

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""

    try:
        service_id = int(data.split(":", 1)[1])
    except Exception:
        await send_or_edit(update, "Invalid service selection.", InlineKeyboardMarkup(footer()))
        return ConversationHandler.END


    # Clean up any previous transient order messages from an earlier attempt
    await _cleanup_temp_order_messages(update, context)

    # Reset any prior order state (do not touch menu navigation selections)
    for k in ("order_service_id", "order_link", "order_qty", "order_confirm_message_id", "summary_msg_id", "error_msg_id"):
        context.user_data.pop(k, None)

    context.user_data["order_service_id"] = service_id

    # Build & send a Service Detail Card (do NOT edit or delete the Service List message)
    Session = context.application.bot_data["Session"]

    platform = "N/A"
    subtype = "Standard"
    name = "N/A"
    rate_disp = "N/A"
    start_disp = "N/A"
    refill_disp = "N/A"

    try:
        async with Session() as session:
            svc = (await session.execute(select(Service).where(Service.id == service_id))).scalars().first()
            if not svc:
                await update.effective_chat.send_message("Service not found. Please select again.")
                return ConversationHandler.END

            platform = (svc.platform or "").strip() or "N/A"
            subtype = ((svc.sub_type or "Standard").strip() or "Standard")
            name = (svc.name or "").strip() or "N/A"
            start_disp = (getattr(svc, "start_time", None) or "").strip() or "N/A"
            refill_disp = (getattr(svc, "refill", None) or "").strip() or "N/A"

            # Compute the displayed rate (per 1k) using cheapest vendor mapping + effective markup
            try:
                global_markup = float(await get_markup_percent(session))
            except Exception:
                global_markup = float(context.application.bot_data.get("markup_percent", 10.0) or 10.0)

            try:
                vs, _vendor = await pick_cheapest_vendor(session, svc.id)
                effective_markup = float(svc.custom_markup) if svc.custom_markup is not None else float(global_markup)
                display_rate = apply_markup(float(vs.vendor_rate), effective_markup)
                rate_disp = f"{float(display_rate):.2f}"
            except Exception:
                # Keep N/A if pricing cannot be computed at this moment
                rate_disp = "N/A"
    except Exception:
        pass

    card = (
        f"✅ Selected: {platform} [{subtype}] - {name}\n"
        f"💰 Rate: ${rate_disp}/k\n"
        f"⏱️ Start: {start_disp} | 🔄 Refill: {refill_disp}\n\n"
        "Please send the link/username to proceed."
    )

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("❌ Cancel", callback_data="ordcfm:cancel")],
        ]
    )

    msg = await update.effective_chat.send_message(card, reply_markup=kb, disable_web_page_preview=True)
    context.user_data["detail_msg_id"] = int(getattr(msg, "message_id", 0) or 0) or None

    return ORDER_CONV_LINK

async def order_conv_link_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 2: Link Input -> validate and ask for Quantity."""
    if await _maintenance_blocker(update, context):
        return ConversationHandler.END

    # Track the user's link message for later cleanup
    if update.message:
        context.user_data["user_link_msg_id"] = int(getattr(update.message, "message_id", 0) or 0) or None

    link = (update.message.text or "").strip() if update.message else ""
    if not _is_valid_order_link(link):
        await update.effective_message.reply_text("Invalid link/URL. Please send a valid URL (https://...) or username.")
        return ORDER_CONV_LINK

    context.user_data["order_link"] = link

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("❌ Cancel", callback_data="ordcfm:cancel")],
        ]
    )
    msg = await update.effective_message.reply_text("Send quantity (number):", reply_markup=kb, disable_web_page_preview=True)
    context.user_data["qty_request_msg_id"] = int(getattr(msg, "message_id", 0) or 0) or None
    return ORDER_CONV_QTY


async def order_conv_qty_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 3: Quantity Input -> validate bounds, compute charge, send NEW confirmation message."""
    if await _maintenance_blocker(update, context):
        return ConversationHandler.END

    # Track the user's quantity message for later cleanup
    if update.message:
        context.user_data["user_qty_msg_id"] = int(getattr(update.message, "message_id", 0) or 0) or None

    txt = (update.message.text or "").strip() if update.message else ""
    try:
        qty = int(txt)
    except Exception:
        await update.effective_message.reply_text("Invalid quantity. Send a whole number.")
        return ORDER_CONV_QTY

    service_id = int(context.user_data.get("order_service_id", 0) or 0)
    link = str(context.user_data.get("order_link", "") or "")
    if not service_id or not link:
        await update.effective_message.reply_text("Order state missing. Please select a service again.")
        return ConversationHandler.END

    Session = context.application.bot_data["Session"]

    async with Session() as session:
        svc = (await session.execute(select(Service).where(Service.id == service_id))).scalars().first()
        if not svc:
            await update.effective_message.reply_text("Service not found. Please select again.")
            return ConversationHandler.END

        # Validate against service bounds
        try:
            min_q = int(svc.min or 0)
        except Exception:
            min_q = 0
        max_q = None
        try:
            if svc.max is not None:
                max_q = int(svc.max)
        except Exception:
            max_q = None

        if qty <= 0 or (min_q and qty < min_q) or (max_q is not None and qty > max_q):
            lim = f"{min_q} - {max_q}" if max_q is not None else f"{min_q}+"
            await update.effective_message.reply_text(f"Invalid quantity. Allowed: {lim}.")
            return ORDER_CONV_QTY

        # Compute display rate using cheapest vendor mapping + effective markup
        try:
            global_markup = float(await get_markup_percent(session))
        except Exception:
            global_markup = float(context.application.bot_data.get("markup_percent", 10.0) or 10.0)

        try:
            vs, vendor = await pick_cheapest_vendor(session, svc.id)
        except RuntimeError:
            await update.effective_message.reply_text('❌ This service is temporarily unavailable . Please pick another service.')
            settings = context.application.bot_data.get('settings') or load_settings()
            alert_text = f"🚨 CRITICAL: No active vendor mapping for Service ID {svc.id}. Please fix immediately."
            for admin_id in (getattr(settings, 'admin_ids', None) or []):
                try:
                    await context.application.bot.send_message(chat_id=admin_id, text=alert_text)
                except Exception:
                    pass
            return ConversationHandler.END
        effective_markup = float(svc.custom_markup) if svc.custom_markup is not None else float(global_markup)
        # Compute rates/costs
        vendor_rate = float(vs.vendor_rate)
        vendor_cost = round(calc_charge(vendor_rate, qty), 4)  # base cost (no markup)

        display_rate = apply_markup(vendor_rate, effective_markup)
        charge = round(calc_charge(display_rate, qty), 4)

        # Enforce minimum order value ($1.00) based on USER CHARGE (includes markup).
        # IMPORTANT: If we enforce this on vendor_cost, services with vendor_rate < $1/1k but high markup
        # (e.g., $0.90/k vendor -> $1.80/k displayed) will be incorrectly rejected even when the user pays >= $1.
        # Admins are exempt from the $1.00 minimum order rule (ADMIN_IDS from .env).
        settings = context.application.bot_data.get('settings') or load_settings()
        admin_ids = (getattr(settings, 'admin_ids', None) or [])
        try:
            admin_ids = {int(x) for x in admin_ids}
        except Exception:
            admin_ids = set()
        admin_user = bool(update.effective_user and int(update.effective_user.id) in admin_ids)

        if (not admin_user) and float(charge) < 1.00:
            # Estimate the minimum quantity needed at the current (display) rate. Rates are per 1000 units.
            min_qty_needed = None
            try:
                per_unit = float(display_rate) / 1000.0
                if per_unit > 0:
                    min_qty_needed = int(math.ceil(1.00 / per_unit))
            except Exception:
                min_qty_needed = None

            extra = f"\nMinimum quantity for this service at current price: {min_qty_needed:,}." if min_qty_needed else ""
            msg = await update.effective_message.reply_text(
                "❌ Order value is below $1.00. Minimum order is $1.00. Please increase quantity." + extra
            )
            context.user_data["error_msg_id"] = int(getattr(msg, "message_id", 0) or 0) or None
            return ORDER_CONV_QTY

    context.user_data["order_qty"] = qty
    # Step 3 MUST be a NEW message (do not edit)
    svc_platform = (svc.platform or "").strip()
    svc_quality = ((svc.sub_type or "Standard").strip() or "Standard")
    svc_name = (svc.name or "").strip()
    service_label = f"{svc_platform} [{svc_quality}] - {svc_name}"

    speed_raw = (getattr(svc, "speed", None) or "").strip()
    refill_raw = (getattr(svc, "refill", None) or "").strip()
    start_time_raw = (getattr(svc, "start_time", None) or "").strip()

    # Specs line MUST always be visible (show N/A when missing)
    speed_disp = html.escape(speed_raw) if speed_raw else "N/A"
    refill_disp = html.escape(refill_raw) if refill_raw else "N/A"
    start_disp = html.escape(start_time_raw) if start_time_raw else "N/A"
    specs_line = f"🔹 <b>Specs:</b> Speed: {speed_disp} | Refill: {refill_disp} | Start: {start_disp}\n"

    total_str = f"{float(charge):.2f}"
    summary = (
        "🧾 <b>Order Summary</b>\n"
        f"🔹 <b>Service:</b> {html.escape(service_label)}\n"
        f"{specs_line}"
        f"🔹 <b>Cost:</b> ${html.escape(total_str)} | Qty: {qty}\n"
        "🔹 <b>Status:</b> Pending\n"
        f"🔹 <b>Link:</b> <code>{html.escape(str(link or ''))}</code>\n\n"
        "Confirm this order?"
    )

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✅ Confirm", callback_data="ordcfm:confirm"), InlineKeyboardButton("❌ Cancel", callback_data="ordcfm:cancel")],
        ]
    )

    msg = await update.effective_message.reply_text(
        summary,
        reply_markup=kb,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )

    context.user_data["order_confirm_message_id"] = int(getattr(msg, "message_id", 0) or 0) or None
    context.user_data["summary_msg_id"] = context.user_data.get("order_confirm_message_id")


    return ORDER_CONV_CONFIRM


async def order_conv_confirm_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 4: Handle confirmation callback."""
    if await _maintenance_blocker(update, context):
        return ConversationHandler.END

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""

    # Helper: clear state
    def _clear_state() -> None:
        for k in ("order_service_id", "order_link", "order_qty", "order_confirm_message_id", "summary_msg_id", "error_msg_id"):
            context.user_data.pop(k, None)

    # Cancel
    if data == "ordcfm:cancel":
        # Nuclear wipe: remove ALL transient messages for the order flow (including Order Summary)
        await _nuclear_wipe_order_messages(update, context)
        _clear_state()
        # Also clear any extra tracking keys used by the nuclear wipe
        for k in ("summary_msg_id", "error_msg_id"):
            context.user_data.pop(k, None)

        # Reset user back to the Main Menu (no "Cancelled" text)
        await send_main_menu_message(update, context)
        return ConversationHandler.END

    # Confirm
    if data != "ordcfm:confirm":
        return ORDER_CONV_CONFIRM

    service_id = int(context.user_data.get("order_service_id", 0) or 0)
    link = str(context.user_data.get("order_link", "") or "")
    qty = int(context.user_data.get("order_qty", 0) or 0)

    if not service_id or not link or not qty:
        await _cleanup_temp_order_messages(update, context)
        try:
            if q.message:
                await q.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        _clear_state()
        await send_main_menu_message(update, context)
        return ConversationHandler.END

    Session = context.application.bot_data["Session"]

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        async with Session() as session:
            user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)
            svc = (await session.execute(select(Service).where(Service.id == service_id))).scalars().first()
            if not svc:
                raise RuntimeError("Service not found.")

            try:
                markup_percent = float(await get_markup_percent(session))
            except Exception:
                markup_percent = float(context.application.bot_data.get("markup_percent", 10.0) or 10.0)

            order = await create_order(session, user, svc, link, int(qty), float(markup_percent))

            
            # Cleanup: delete interaction messages BEFORE sending receipt (mirror Cancel behavior)
            await _nuclear_wipe_order_messages(update, context)

            receipt_text = _build_order_receipt_text(
                svc,
                float(getattr(order, "charge", 0.0) or 0.0),
                int(getattr(order, "quantity", qty) or qty),
                str(getattr(order, "status", "") or "Pending"),
            )

            receipt_msg = await update.effective_chat.send_message(
                receipt_text,
                disable_web_page_preview=True,
            )

            # Capture receipt message identifiers so we can edit the receipt later (on completion).
            order.receipt_message_id = int(getattr(receipt_msg, "message_id", 0) or 0) or None
            order.receipt_chat_id = int(getattr(receipt_msg, "chat_id", 0) or (update.effective_chat.id if update.effective_chat else 0)) or None
            await commit_with_retry(session)

    except Exception as e:
        # User-facing error
        err = str(e)
        lerr = err.lower()
        if "insufficient balance" in lerr or "quantity must be between" in lerr:
            await update.effective_chat.send_message(f"❌ {err}")
        else:
            await update.effective_chat.send_message("❌ Please try again in a few hours.")

        # Nuclear wipe on failure: remove ALL transient messages (including Order Summary)
        await _nuclear_wipe_order_messages(update, context)
        _clear_state()
        await send_main_menu_message(update, context)
        return ConversationHandler.END

    # Success: interaction messages were already cleaned up (nuclear wipe before receipt)
    _clear_state()
    await send_main_menu_message(update, context)
    return ConversationHandler.END


async def order_conv_fallback_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback cancel handler for inline Cancel buttons during the conversation.

    Nuclear wipe: delete ALL tracked transient messages and reset the user to the Main Menu.
    Does NOT delete Service List menu messages.
    """
    q = update.callback_query
    if q:
        try:
            await _safe_answer(q)
        except Exception:
            pass

    await _nuclear_wipe_order_messages(update, context)

    # Clear only order-related state (preserve navigation selections)
    for k in ("order_service_id", "order_link", "order_qty", "order_confirm_message_id", "summary_msg_id", "error_msg_id"):
        context.user_data.pop(k, None)

    await send_main_menu_message(update, context)
    return ConversationHandler.END


async def ord_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        return
    try:
        oid = int(parts[1])
    except Exception:
        return
    await screen_order_detail(update, context, oid)


async def dep_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""

    parts = data.split(":")
    if len(parts) < 2:
        return

    if data.startswith("dep:coin:"):
        if len(parts) < 3:
            return
        coin = (parts[2] or "").upper()
        context.user_data["deposit_coin"] = coin
        context.user_data["input_mode"] = INP_NONE
        await screen_deposit_amounts(update, context, coin)
        return

    if data.startswith("dep:amt:"):
        if len(parts) < 4:
            await send_or_edit(update, "Invalid amount.", InlineKeyboardMarkup(footer()))
            return
        try:
            _, _, coin, amt_s = data.split(":", 3)
            amt = float(amt_s)
        except Exception:
            await send_or_edit(update, "Invalid amount.", InlineKeyboardMarkup(footer()))
            return
        coin = coin.upper()
        min_amount = 5 if coin == "TRX" else 1
        if amt < min_amount:
            await send_or_edit(update, f"Minimum deposit for {coin} is ${min_amount}.", InlineKeyboardMarkup(footer()))
            return
        await create_invoice(update, context, amt, coin)
        return

    if data.startswith("dep:custom:"):
        if len(parts) < 3:
            return
        coin = (parts[2] or "").upper()
        context.user_data["deposit_coin"] = coin
        context.user_data["input_mode"] = INP_DEPOSIT_CUSTOM
        min_amount = 5 if coin == "TRX" else 1
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="nav:wallet")]] + footer())
        await send_or_edit(update, f"Send deposit amount in USD (min ${min_amount}) for {coin}:", kb)
        return

    # Backward compatibility: dep:<amount> and dep:custom
    if data.startswith("dep:") and data.count(":") == 1:
        if len(parts) < 2:
            return
        arg = parts[1]
        if arg == "custom":
            context.user_data["deposit_coin"] = "USDT"
            context.user_data["input_mode"] = INP_DEPOSIT_CUSTOM
            await send_or_edit(update, "Send deposit amount in USD:", InlineKeyboardMarkup(footer()))
            return
        try:
            amt = float(arg)
        except Exception:
            await send_or_edit(update, "Invalid deposit amount.", InlineKeyboardMarkup(footer()))
            return
        await create_invoice(update, context, amt, "USDT")
        return

async def pay_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        return
    try:
        invoice_id = int(parts[1])
    except Exception:
        return
    await check_payment(update, context, invoice_id)


async def adm_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    cmd = q.data
    Session = context.application.bot_data["Session"]

    if cmd == "adm:maint":
        maint_on = False
        try:
            async with Session() as session:
                cur = await is_maintenance_on(session)
                maint_on = not bool(cur)
                await toggle_maintenance(session, maint_on)
        except Exception:
            # If toggle fails, keep it OFF visually
            maint_on = False

        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Markup %", callback_data="adm:markup"),
                 InlineKeyboardButton("Vendors", callback_data="adm:vendors")],
                [InlineKeyboardButton("Catalog Sync", callback_data="adm:sync"),
                 InlineKeyboardButton("Add Funds", callback_data="adm:addfunds")],
                [InlineKeyboardButton("Set Service Markup", callback_data="adm:svcmarkup")],
            [InlineKeyboardButton("🔧 Menu Manager", callback_data="adm:menumgr")],
                [InlineKeyboardButton(f"⚙️ Maintenance Mode ({'ON' if maint_on else 'OFF'})", callback_data="adm:maint")],
                [InlineKeyboardButton("Daily Stats", callback_data="adm:dailystats"),
                 InlineKeyboardButton("Check Stats", callback_data="adm:vendorstats")],
            ] + footer()
        )
        await send_or_edit(update, "⚙️ Admin Panel", kb)
        return

    if cmd == "adm:markup":
        context.user_data["input_mode"] = INP_MARKUP
        kb = InlineKeyboardMarkup(footer_admin())

        current = 10.0
        try:
            async with Session() as session:
                current = float(await get_markup_percent(session))
        except Exception:
            current = 10.0

        await send_or_edit(update, f"Current Global Markup: {current}%\n\nEnter new value to update:", kb)
        return

    if cmd == "adm:vendors":
        await screen_admin_vendors(update, context)
        return

    if cmd == "adm:sync":
        # Run sync in the background so the bot doesn't freeze for other users.
        nav_set(context, ADMIN_SYNC_RESULT, {}, push=True)

        kb = InlineKeyboardMarkup(footer_admin())
        await send_or_edit(update, "🔄 Sync started in background...\\n\\nI'll update this message when it's done.", kb)

        try:
            chat_id = int(update.effective_chat.id)
            message_id = int(q.message.message_id) if q and q.message else int(update.effective_message.message_id)
            try:
                context.application.create_task(_admin_catalog_sync_task(context.application, chat_id, message_id))
            except Exception:
                asyncio.create_task(_admin_catalog_sync_task(context.application, chat_id, message_id))
        except Exception:
            pass
        return

    if cmd == "adm:addfunds":
        context.user_data["input_mode"] = INP_ADDFUNDS_UID
        kb = InlineKeyboardMarkup(footer_admin())
        await send_or_edit(update, "Send target Telegram User ID:", kb)
        return

    if cmd == "adm:svcmarkup":
        # Admin: Set Service Markup (by Platform -> Category)
        context.user_data["input_mode"] = INP_NONE
        context.user_data.pop("svcmarkup_service_id", None)
        context.user_data.pop("svcmarkup_platform", None)
        context.user_data.pop("svcmarkup_category", None)

        # Build platform buttons
        platforms = list(PLATFORM_LIST) if PLATFORM_LIST else []
        if not platforms:
            # Fallback: derive from DB if PLATFORM_LIST is empty
            Session = context.application.bot_data["Session"]
            async with Session() as session:
                platforms = (await session.execute(
                    select(Service.platform).distinct().order_by(Service.platform.asc())
                )).scalars().all()
            platforms = [p for p in platforms if p]

        context.user_data["svcmarkup_platforms"] = platforms
        buttons = [
            InlineKeyboardButton(p, callback_data=f"svcmarkup:plat:{i}")
            for i, p in enumerate(platforms)
        ]
        kb = InlineKeyboardMarkup(rows2(buttons) + footer_admin())
        await send_or_edit(update, "Select platform to set markup for:", kb)
        return

    if cmd == "adm:menumgr":
        context.user_data["mm_page"] = 0
        await screen_menu_manager(update, context)
        return

    if cmd.startswith("adm:dailystats"):
        # Date navigation for Daily Stats
        if cmd == "adm:dailystats":
            await screen_daily_stats(update, context)
            return
        if cmd in ("adm:dailystats:prev", "adm:dailystats:next"):
            sel = str(context.user_data.get("admin_daily_stats_date") or "")
            try:
                y, m, d = [int(x) for x in sel.split("-")]
                cur = datetime(y, m, d, tzinfo=timezone.utc)
            except Exception:
                now = datetime.now(timezone.utc)
                cur = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)

            delta = -1 if cmd.endswith(":prev") else 1
            nxt = cur + timedelta(days=delta)
            context.user_data["admin_daily_stats_date"] = nxt.strftime("%Y-%m-%d")
            await screen_daily_stats(update, context)
            return

    if cmd == "adm:vendorstats":
        await screen_vendor_stats(update, context)
        return

    if cmd == "adm:addvendor":
        context.user_data["input_mode"] = INP_ADDV_NAME
        kb = InlineKeyboardMarkup(footer_admin())
        await send_or_edit(update, "Add Vendor — send Vendor Name:", kb)
        return

    if cmd == "adm:testvendor":
        await screen_vendor_list(update, context, mode="test")
        return

    if cmd == "adm:togglevendor":
        await screen_vendor_list(update, context, mode="toggle")
        return




async def mm_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menu Manager callbacks: toggle visibility and paginate."""
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    data = q.data or ""
    if data == "mm:page:prev":
        context.user_data["mm_page"] = int(context.user_data.get("mm_page") or 0) - 1
        await screen_menu_manager(update, context)
        return
    if data == "mm:page:next":
        context.user_data["mm_page"] = int(context.user_data.get("mm_page") or 0) + 1
        await screen_menu_manager(update, context)
        return

    if data.startswith("mm:nop:"):
        return

    if data.startswith("mm:t:"):
        try:
            idx = int(data.split(":", 2)[2])
        except Exception:
            await screen_menu_manager(update, context)
            return

        combos = context.user_data.get("mm_combos") or []
        if idx < 0 or idx >= len(combos):
            await screen_menu_manager(update, context)
            return

        platform, category = combos[idx]
        Session = context.application.bot_data["Session"]
        async with Session() as session:
            n = (await session.execute(
                select(func.count(Service.id))
                .where(Service.platform == platform, Service.category == category, Service.is_active == True)
            )).scalar_one()
            n = int(n or 0)

            mv = (await session.execute(
                select(MenuVisibility)
                .where(MenuVisibility.platform == platform, MenuVisibility.category == category)
                .limit(1)
            )).scalar_one_or_none()

            default_visible = True if n > 0 else False
            if mv is None:
                mv = MenuVisibility(platform=platform, category=category, is_visible=(not default_visible))
                session.add(mv)
            else:
                mv.is_visible = not bool(mv.is_visible)

            try:
                await session.commit()
            except Exception:
                await session.rollback()

        await screen_menu_manager(update, context)
        return
async def vend_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    if not is_admin(update, context):
        kb = InlineKeyboardMarkup(footer())
        await send_or_edit(update, "Not authorized.", kb)
        return

    data = q.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        return
    try:
        vendor_id = int(parts[1])
    except Exception:
        return
    mode = context.user_data.get("vendor_pick_mode")
    Session = context.application.bot_data["Session"]

    if mode == "test":
        async with Session() as session:
            _ok, msg = await test_vendor(session, vendor_id)
        kb = InlineKeyboardMarkup(footer_admin())
        await send_or_edit(update, f"Test result: {msg}", kb)
        return

    if mode == "toggle":
        async with Session() as session:
            v = await toggle_vendor(session, vendor_id)
        kb = InlineKeyboardMarkup(footer_admin())
        await send_or_edit(update, f"{v.name} is now {'ON' if v.is_active else 'OFF'}", kb)
        return

    kb = InlineKeyboardMarkup(footer_admin())
    await send_or_edit(update, "Unknown vendor action.", kb)




async def svcmarkup_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    """Admin: Set Service Markup via Platform -> Category -> % (bulk update)."""
    q = update.callback_query
    await _safe_answer(q)
    data = q.data or ""

    if not is_admin(update, context):
        await send_or_edit(update, "Not authorized.", InlineKeyboardMarkup(footer_admin()))
        return

    parts = data.split(":")
    if len(parts) < 2:
        return

    action = parts[1]

    async def _render_platforms() -> None:
        platforms = context.user_data.get("svcmarkup_platforms") or list(PLATFORM_LIST) if PLATFORM_LIST else []
        if not platforms:
            Session = context.application.bot_data["Session"]
            async with Session() as session:
                platforms = (await session.execute(
                    select(Service.platform).distinct().order_by(Service.platform.asc())
                )).scalars().all()
            platforms = [p for p in platforms if p]
        context.user_data["svcmarkup_platforms"] = platforms
        buttons = [InlineKeyboardButton(p, callback_data=f"svcmarkup:plat:{i}") for i, p in enumerate(platforms)]
        kb = InlineKeyboardMarkup(rows2(buttons) + footer_admin())
        await send_or_edit(update, "Select platform to set markup for:", kb)

    async def _render_categories(plat: str) -> None:
        Session = context.application.bot_data["Session"]
        async with Session() as session:
            cats = (await session.execute(
                select(Service.category)
                .where(Service.platform == plat)
                .distinct()
                .order_by(Service.category.asc())
            )).scalars().all()
        cats = [c for c in cats if c]
        context.user_data["svcmarkup_categories"] = cats
        buttons = [InlineKeyboardButton(c, callback_data=f"svcmarkup:cat:{i}") for i, c in enumerate(cats)]
        kb_rows = rows2(buttons)
        kb_rows.append([InlineKeyboardButton("🔙 Back", callback_data="svcmarkup:backplat")])
        kb_rows += footer_admin()
        kb = InlineKeyboardMarkup(kb_rows)
        await send_or_edit(update, f"Select category for {plat}:", kb)


    async def _render_subtypes(plat: str, cat: str) -> None:
        Session = context.application.bot_data["Session"]
        async with Session() as session:
            subs_all = (await session.execute(
                select(Service.sub_type)
                .where(Service.platform == plat, Service.category == cat)
                .distinct()
                .order_by(Service.sub_type.asc())
            )).scalars().all()

        has_standard = any((s is None) or (isinstance(s, str) and s.strip() == "") for s in subs_all)

        # Keep only non-empty explicit sub_type values
        subs_clean = []
        for s in subs_all:
            if s is None:
                continue
            if isinstance(s, str) and s.strip() == "":
                continue
            subs_clean.append(str(s))

        # De-dup while preserving order
        seen = set()
        subs_clean = [s for s in subs_clean if not (s in seen or seen.add(s))]

        subs_labels: list[str] = []
        subs_raw: list[str] = []

        if has_standard:
            subs_labels.append("Standard")
            subs_raw.append("")

        for s in subs_clean:
            subs_labels.append(s)
            subs_raw.append(s)

        context.user_data["svcmarkup_subtypes"] = subs_labels
        context.user_data["svcmarkup_subtypes_raw"] = subs_raw

        if not subs_labels:
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="svcmarkup:backcats")],
            ] + footer_admin())
            await send_or_edit(update, f"No qualities found for {plat} - {cat}.", kb)
            return

        buttons = [InlineKeyboardButton(label, callback_data=f"svcmarkup:sub:{i}") for i, label in enumerate(subs_labels)]
        kb_rows = rows2(buttons)
        kb_rows.append([InlineKeyboardButton("🔙 Back", callback_data="svcmarkup:backcats")])
        kb_rows += footer_admin()
        kb = InlineKeyboardMarkup(kb_rows)
        await send_or_edit(update, f"Select quality for {plat} - {cat}:", kb)

    if action == "backplat":
        context.user_data.pop("svcmarkup_platform", None)
        context.user_data.pop("svcmarkup_category", None)
        context.user_data.pop("svcmarkup_categories", None)
        await _render_platforms()
        return

    if action == "plat":
        if len(parts) < 3:
            await _render_platforms()
            return
        try:
            idx = int(parts[2])
        except Exception:
            await _render_platforms()
            return

        platforms = context.user_data.get("svcmarkup_platforms") or list(PLATFORM_LIST)
        if idx < 0 or idx >= len(platforms):
            await _render_platforms()
            return

        plat = platforms[idx]
        context.user_data["svcmarkup_platform"] = plat
        context.user_data.pop("svcmarkup_category", None)
        await _render_categories(plat)
        return

    if action == "cat":
        if len(parts) < 3:
            return
        try:
            idx = int(parts[2])
        except Exception:
            return

        plat = context.user_data.get("svcmarkup_platform")
        cats = context.user_data.get("svcmarkup_categories") or []
        if not plat or idx < 0 or idx >= len(cats):
            # Re-render categories if something is missing
            if plat:
                await _render_categories(str(plat))
            else:
                await _render_platforms()
            return

        cat = cats[idx]
        context.user_data["svcmarkup_category"] = cat
        context.user_data.pop("svcmarkup_subtype", None)
        context.user_data.pop("svcmarkup_subtypes", None)
        context.user_data["input_mode"] = INP_SVCMARKUP_SUBTYPE

        await _render_subtypes(str(plat), str(cat))
        return

    if action == "sub":
        if len(parts) < 3:
            return
        try:
            idx = int(parts[2])
        except Exception:
            return
        plat = context.user_data.get("svcmarkup_platform")
        cat = context.user_data.get("svcmarkup_category")
        labels = context.user_data.get("svcmarkup_subtypes") or []
        subs = context.user_data.get("svcmarkup_subtypes_raw") or labels
        if not plat or not cat or idx < 0 or idx >= len(subs):
            # Re-render if something is missing
            if plat and cat:
                await _render_subtypes(str(plat), str(cat))
            elif plat:
                await _render_categories(str(plat))
            else:
                await _render_platforms()
            return

        sub = subs[idx]
        context.user_data["svcmarkup_subtype"] = sub
        if idx < len(labels):
            context.user_data["svcmarkup_subtype_label"] = labels[idx]
        context.user_data["input_mode"] = INP_SVCMARKUP_PCT

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back", callback_data="svcmarkup:backsub")],
        ] + footer_admin())

        Session = context.application.bot_data["Session"]
        async with Session() as session:
            # "Standard" maps to NULL/empty sub_type in the DB
            if (sub is None) or (isinstance(sub, str) and sub.strip() == ""):
                existing = (await session.execute(
                    text(
                        "SELECT custom_markup FROM services "
                        "WHERE platform = :plat AND category = :cat "
                        "AND (sub_type IS NULL OR sub_type = '') "
                        "AND custom_markup IS NOT NULL "
                        "LIMIT 1"
                    ),
                    {"plat": str(plat), "cat": str(cat)},
                )).scalar_one_or_none()
            else:
                existing = (await session.execute(
                    select(Service.custom_markup)
                    .where(
                        Service.platform == str(plat),
                        Service.category == str(cat),
                        Service.sub_type == str(sub),
                        Service.custom_markup.is_not(None),
                    )
                    .limit(1)
                )).scalar_one_or_none()

            current = existing if existing is not None else await get_markup_percent(session)

        try:
            current_str = str(int(float(current))) if float(current).is_integer() else str(current)
        except Exception:
            current_str = str(current)

        sub_label = context.user_data.get("svcmarkup_subtype_label") or (str(sub) if sub else "Standard")

        prompt = f"Current Markup for {plat} - {cat} - {sub_label}: {current_str}%\n\nEnter new value to update:"
        await send_or_edit(update, prompt, kb)
        return

    if action == "backsub":
        plat = context.user_data.get("svcmarkup_platform")
        cat = context.user_data.get("svcmarkup_category")
        if plat and cat:
            context.user_data.pop("svcmarkup_subtype", None)
            context.user_data["input_mode"] = INP_SVCMARKUP_SUBTYPE
            await _render_subtypes(str(plat), str(cat))
        elif plat:
            await _render_categories(str(plat))
        else:
            await _render_platforms()
        return

    if action == "backcats":
        plat = context.user_data.get("svcmarkup_platform")
        if plat:
            context.user_data.pop("svcmarkup_category", None)
            context.user_data.pop("svcmarkup_subtype", None)
            context.user_data.pop("svcmarkup_subtypes", None)
            context.user_data["input_mode"] = INP_NONE
            await _render_categories(str(plat))
        else:
            await _render_platforms()
        return


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    mode = context.user_data.get("input_mode", INP_NONE)

    if mode == INP_NONE:
        return

    # -----------------------------
    # Order flow (Confirm -> Link -> Qty -> Create)
    # -----------------------------
    if mode == INP_ORDER_LINK:
        link = (update.message.text or "").strip()
        if not link or len(link) < 3:
            await update.message.reply_text("Invalid link, try again.")
            return

        context.user_data["order_link"] = link
        context.user_data["input_mode"] = INP_ORDER_QTY

        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("❌ Cancel", callback_data="oc:cancel")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
            ]
        )
        await update.message.reply_text("Send quantity (number):", reply_markup=kb)
        return

    if mode == INP_ORDER_QTY:
        txt = (update.message.text or "").strip()

        # Strict: if input is NOT a number, delete the message and reset to main menu.
        try:
            qty = int(txt)
        except Exception:
            try:
                await update.message.delete()
            except Exception:
                pass
            for k in ("order_service_id", "order_vendor_name", "order_price_display", "order_link", "order_qty", "failed_attempts", "last_qty_error_msg_id"):
                context.user_data.pop(k, None)
            context.user_data["input_mode"] = INP_NONE
            await send_main_menu_message(update, context)
            return

        service_id = int(context.user_data.get("order_service_id", 0) or 0)
        link = str(context.user_data.get("order_link", ""))
        markup = float(context.application.bot_data.get("markup_percent", 10.0) or 10.0)
        Session = context.application.bot_data["Session"]

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        async with Session() as session:
            user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)
            svc = (await session.execute(select(Service).where(Service.id == service_id))).scalars().first()
            if not svc:
                await update.message.reply_text("Service not found.")
                context.user_data["input_mode"] = INP_NONE
                return

            # Strict quantity bounds enforcement with 2-attempt lockout.
            try:
                min_q = int(svc.min or 0)
            except Exception:
                min_q = 0
            max_q = None
            try:
                if svc.max is not None:
                    max_q = int(svc.max)
            except Exception:
                max_q = None

            invalid = (qty <= 0) or (qty < min_q) or (max_q is not None and qty > max_q)
            if invalid:
                MAX_ATTEMPTS = 2
                attempts = int(context.user_data.get("failed_attempts", 0) or 0) + 1
                context.user_data["failed_attempts"] = attempts

                # First failure: keep it simple, no "attempts left" text.
                if attempts < MAX_ATTEMPTS:
                    try:
                        err = await update.message.reply_text("Invalid quantity. Please enter a valid number.")
                        context.user_data["last_qty_error_msg_id"] = err.message_id
                    except Exception as e:
                        logger.exception(f"Error sending invalid quantity message: {e}")
                    return

                # Second failure: clean up (delete user's failed number + bot's prior error), reset to main menu.
                try:
                    await update.message.delete()
                except Exception:
                    pass

                err_id = context.user_data.get("last_qty_error_msg_id")
                if err_id:
                    try:
                        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=int(err_id))
                    except Exception:
                        pass

                for k in (
                    "order_service_id",
                    "order_vendor_name",
                    "order_price_display",
                    "order_link",
                    "order_qty",
                    "failed_attempts",
                    "last_qty_error_msg_id",
                ):
                    context.user_data.pop(k, None)
                context.user_data["input_mode"] = INP_NONE
                await send_main_menu_message(update, context)
                return

            # Reset failed attempts / error tracking on success
            context.user_data.pop("failed_attempts", None)
            context.user_data.pop("last_qty_error_msg_id", None)
            try:
                # Create and persist the order immediately (do not wait for vendor Website ID)
                vs, vendor = await pick_cheapest_vendor(session, svc.id)

                if qty < int(vs.vendor_min) or qty > int(vs.vendor_max):
                    raise RuntimeError(f"Quantity must be between {vs.vendor_min} and {vs.vendor_max} for this service.")

                effective_markup = float(svc.custom_markup) if svc.custom_markup is not None else float(markup)
                display_rate = apply_markup(float(vs.vendor_rate), effective_markup)
                charge = round(calc_charge(display_rate, qty), 4)

                cost = round((float(vs.vendor_rate) / 1000.0) * float(qty), 4)
                profit = round(charge - cost, 4)

                if float(user.balance) < charge:
                    raise RuntimeError(f"Insufficient balance. Need {charge:.4f}, you have {float(user.balance):.4f}")

                # Deduct balance and create order with Pending status
                user.balance = float(user.balance) - charge
                await session.flush()

                order = Order(
                    user_id=user.id,
                    service_id=svc.id,
                    link=link,
                    quantity=qty,
                    charge=charge,
                    cost=cost,
                    profit=profit,
                    status="Pending",
                    vendor_name=vendor.name,
                    vendor_order_id=None,
                )
                session.add(order)
                await commit_with_retry(session)
                await session.refresh(order)

                async def _submit_vendor_order(
                    order_id: int,
                    vendor_id: int,
                    vendor_service_id: str,
                    link_: str,
                    qty_: int,
                    charge_: float,
                    user_chat_id: int,
                ) -> None:
                    # Place the vendor order in the background; update Order ID later
                    async with async_session() as s2:
                        o = (await s2.execute(select(Order).where(Order.id == order_id))).scalars().first()
                        if not o:
                            return

                        v = (await s2.execute(select(Vendor).where(Vendor.id == vendor_id))).scalars().first()
                        if not v:
                            u = (await s2.execute(select(User).where(User.id == o.user_id))).scalars().first()
                            if u:
                                u.balance = float(u.balance) + float(charge_)
                            o.status = "Failed"
                            await commit_with_retry(s2)
                            return

                        adapter = SMMv1Adapter(v)
                        try:
                            website_id = await adapter.add_order(vendor_service_id, link=link_, quantity=qty_)
                            o.vendor_order_id = str(website_id)
                            o.status = "Processing"
                            await commit_with_retry(s2)
                        except Exception:
                            u = (await s2.execute(select(User).where(User.id == o.user_id))).scalars().first()
                            if u:
                                u.balance = float(u.balance) + float(charge_)
                            o.status = "Failed"
                            await commit_with_retry(s2)

                            # Notify user generically
                            try:
                                await context.application.bot.send_message(
                                    chat_id=user_chat_id,
                                    text="❌ Please try again in a few hours.",
                                )
                            except Exception:
                                pass

                            # Notify admins
                            try:
                                settings = context.application.bot_data["settings"]
                                alert_text = (
                                    f"⚠️ Vendor Alert: {v.name} balance insufficient or API failure. "
                                    f"Please top up or check status."
                                )
                                for admin_id in settings.admin_ids:
                                    try:
                                        await context.application.bot.send_message(chat_id=admin_id, text=alert_text)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                try:
                    context.application.create_task(
                        _submit_vendor_order(
                            order.id,
                            vendor.id,
                            str(vs.vendor_service_id),
                            link,
                            int(qty),
                            float(charge),
                            int(update.effective_user.id),
                        )
                    )
                except Exception:
                    asyncio.create_task(
                        _submit_vendor_order(
                            order.id,
                            vendor.id,
                            str(vs.vendor_service_id),
                            link,
                            int(qty),
                            float(charge),
                            int(update.effective_user.id),
                        )
                    )

            except Exception as e:
                err = str(e)
                lerr = err.lower()

                # User-facing errors (keep useful details)
                if ("insufficient balance" in lerr) or ("quantity must be between" in lerr):
                    await update.message.reply_text(f"❌ {err}")
                else:
                    # Vendor-side failure: show generic message to user
                    await update.message.reply_text("❌ Please try again in a few hours.")

                    # Notify admins (vendor balance insufficient or generic API failure)
                    vendor_name = "Unknown"
                    m1 = re.search(r"Vendor failure at ([^:]+):", err)
                    if m1:
                        vendor_name = m1.group(1).strip()
                    else:
                        m2 = re.search(r"Order failed at vendor:\s*(.+)$", err)
                        if m2:
                            vendor_name = m2.group(1).strip()

                    try:
                        settings = context.application.bot_data["settings"]
                        alert_text = (
                            f"⚠️ Vendor Alert: {vendor_name} balance insufficient or API failure. "
                            f"Please top up or check status."
                        )
                        for admin_id in settings.admin_ids:
                            try:
                                await context.application.bot.send_message(chat_id=admin_id, text=alert_text)
                            except Exception:
                                pass
                    except Exception:
                        pass

                context.user_data["input_mode"] = INP_NONE
                return

        for k in ("order_service_id", "order_vendor_name", "order_price_display", "order_link", "order_qty"):
            context.user_data.pop(k, None)
        context.user_data["input_mode"] = INP_NONE

                # Receipt (clean format; editable later)
        receipt_text = _build_order_receipt_text(
            svc,
            float(getattr(order, "charge", 0.0) or 0.0),
            int(getattr(order, "quantity", qty) or qty),
            str(getattr(order, "status", "") or "Pending"),
        )
        receipt_msg = await update.message.reply_text(receipt_text, disable_web_page_preview=True)
        # Persist receipt message identifiers (use a fresh session because the create-order session is closed)
        try:
            async with async_session() as s3:
                o3 = (await s3.execute(select(Order).where(Order.id == order.id))).scalars().first()
                if o3:
                    o3.receipt_message_id = int(getattr(receipt_msg, "message_id", 0) or 0) or None
                    o3.receipt_chat_id = int(getattr(receipt_msg, "chat_id", 0) or (update.effective_chat.id if update.effective_chat else 0)) or None
                    await commit_with_retry(s3)
        except Exception:
            pass
        await send_main_menu_message(update, context)
        return

    # -----------------------------
    # Deposit custom amount (USD)
    # -----------------------------
    
    # -----------------------------
    # Check Order Status (User)
    # -----------------------------
    if mode == INP_CHECK_ORDER:
        txt = (update.message.text or "").strip()

        Session = context.application.bot_data["Session"]
        async with Session() as session:
            user = await get_or_create_user(session, update.effective_user.id, update.effective_user.username)

            order = None
            svc_name = None

            # Search 1: Internal Order ID (integer)
            order_id = None
            if txt.isdigit():
                try:
                    order_id = int(txt)
                except Exception:
                    order_id = None

            if order_id is not None:
                stmt = select(Order).where(Order.id == order_id)
                if not is_admin(update, context):
                    stmt = stmt.where(Order.user_id == user.id)
                order = (await session.execute(stmt)).scalars().first()

            # Search 2: Website/Vendor Order ID (string)
            if not order:
                stmt = select(Order).where(Order.vendor_order_id == txt)
                if not is_admin(update, context):
                    stmt = stmt.where(Order.user_id == user.id)
                order = (await session.execute(stmt)).scalars().first()

            if order:
                svc = (await session.execute(select(Service).where(Service.id == order.service_id))).scalars().first()
                if svc:
                    svc_name = svc.name

        context.user_data["input_mode"] = INP_NONE

        if not order:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
            await update.message.reply_text("Order not found.", reply_markup=kb)
            return

        website_id = order.vendor_order_id or "—"
        service_name = svc_name or f"Service #{order.service_id}"
        vendor_name = order.vendor_name or "—"
        lines = [
            "📦 Order Status",
            "",
            f"Order ID: {website_id}",
            f"Internal Order ID: {order.id}",
            f"Vendor: {vendor_name}",
            f"Service: {service_name}",
            f"Status: {order.status}",
            f"Link: {order.link}",
            f"Quantity: {order.quantity}",
            f"Total Cost: {money(float(order.charge))}",
        ]
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("📦 My Orders", callback_data="nav:orders"),
                 InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
            ]
        )
        await update.message.reply_text("\n".join(lines), reply_markup=kb)
        return

    if mode == INP_DEPOSIT_CUSTOM:
        txt = (update.message.text or "").strip()
        try:
            amount = float(txt)
            if amount <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Invalid amount. Send a number like 10:")
            return

        coin = str(context.user_data.get("deposit_coin", "USDT")).upper()
        min_amount = 5 if coin == "TRX" else 1
        if amount < min_amount:
            await update.message.reply_text(f"Minimum deposit for {coin} is ${min_amount}. Send a higher amount:")
            return

        context.user_data["input_mode"] = INP_NONE
        await create_invoice(update, context, amount, coin)
        return


    # -----------------------------
    # Admin: Add Funds flow (UID -> Amount)
    # -----------------------------
    if mode == INP_ADDFUNDS_UID:
        if not is_admin(update, context):
            context.user_data["input_mode"] = INP_NONE
            await update.message.reply_text("Not authorized.")
            return

        txt = (update.message.text or "").strip()
        try:
            target_id = int(txt)
            if target_id <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Invalid Telegram User ID. Send a numeric ID like 123456789:")
            return

        context.user_data["addfunds_target_id"] = target_id
        context.user_data["input_mode"] = INP_ADDFUNDS_AMT

        kb = InlineKeyboardMarkup(footer_admin())
        await update.message.reply_text("Enter amount:", reply_markup=kb)
        return

    if mode == INP_ADDFUNDS_AMT:
        if not is_admin(update, context):
            context.user_data["input_mode"] = INP_NONE
            await update.message.reply_text("Not authorized.")
            return

        txt = (update.message.text or "").strip()
        try:
            amount = float(txt)
            if amount <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Invalid amount. Send a number like 10:")
            return

        target_id = int(context.user_data.get("addfunds_target_id") or 0)
        if target_id <= 0:
            context.user_data["input_mode"] = INP_NONE
            await update.message.reply_text("Missing target user ID. Start again: Admin → Add Funds.")
            return

        Session = context.application.bot_data["Session"]
        async with Session() as session:
            user = await add_funds(session, target_id, amount)

        context.user_data["input_mode"] = INP_NONE
        context.user_data.pop("addfunds_target_id", None)

        kb = InlineKeyboardMarkup(footer_admin())
        await update.message.reply_text(
            f"✅ Added {money(float(amount))} to {user.telegram_id}. New balance: {money(float(user.balance))}",
            reply_markup=kb,
        )
        return

    # -----------------------------
    # Admin: Set Service Markup (Search -> %)
    # -----------------------------
    if mode == INP_SVCMARKUP_SEARCH:
        if not is_admin(update, context):
            context.user_data["input_mode"] = INP_NONE
            await update.message.reply_text("Not authorized.")
            return

        query = (update.message.text or "").strip()
        if not query:
            await update.message.reply_text("Send Service Name or ID:")
            return

        Session = context.application.bot_data["Session"]
        svc: Optional[Service] = None
        matches: List[Service] = []

        async with Session() as session:
            # Numeric ID lookup
            if query.isdigit():
                sid = int(query)
                svc = (await session.execute(select(Service).where(Service.id == sid))).scalars().first()
            else:
                ql = query.lower()
                matches = (await session.execute(
                    select(Service)
                    .where(func.lower(Service.name).like(f"%{ql}%"))
                    .order_by(Service.id.asc())
                    .limit(6)
                )).scalars().all()
                if len(matches) == 1:
                    svc = matches[0]

        if not svc:
            if matches:
                # Multiple matches — ask for exact ID
                lines = ["Multiple matches found. Send the Service ID:", ""]
                for s in matches[:5]:
                    lines.append(f"- {s.id}: {s.name}")
                kb = InlineKeyboardMarkup(footer_admin())
                await update.message.reply_text("\n".join(lines), reply_markup=kb)
                return
            kb = InlineKeyboardMarkup(footer_admin())
            await update.message.reply_text("Service not found. Send Service Name or ID:", reply_markup=kb)
            return

        context.user_data["svcmarkup_service_id"] = int(svc.id)
        context.user_data["input_mode"] = INP_SVCMARKUP_PCT
        kb = InlineKeyboardMarkup(footer_admin())
        await update.message.reply_text(
            f"Setting markup for: {svc.name}\nSend the percentage (e.g., 30).",
            reply_markup=kb,
        )
        return

    if mode == INP_SVCMARKUP_PCT:
        if not is_admin(update, context):
            context.user_data["input_mode"] = INP_NONE
            await update.message.reply_text("Not authorized.")
            return

        txt = (update.message.text or "").strip()
        try:
            pct = float(txt)
        except Exception:
            await update.message.reply_text("Invalid percent. Send a number like 30:")
            return

        plat = context.user_data.get("svcmarkup_platform")
        cat = context.user_data.get("svcmarkup_category")
        if plat and cat:
            Session = context.application.bot_data["Session"]
            async with Session() as session:
                sub = context.user_data.get("svcmarkup_subtype")
                if (sub is None) or (isinstance(sub, str) and sub.strip() == ""):
                    res = await session.execute(
                        text("UPDATE services SET custom_markup = :markup WHERE platform = :plat AND category = :cat AND (sub_type IS NULL OR sub_type = '')"),
                        {"markup": float(pct), "plat": str(plat), "cat": str(cat)},
                    )
                else:
                    res = await session.execute(
                        text("UPDATE services SET custom_markup = :markup WHERE platform = :plat AND category = :cat AND sub_type = :sub"),
                        {"markup": float(pct), "plat": str(plat), "cat": str(cat), "sub": str(sub)},
                    )
                await commit_with_retry(session)
                updated = int(res.rowcount or 0)

            context.user_data["input_mode"] = INP_NONE
            context.user_data.pop("svcmarkup_platform", None)
            context.user_data.pop("svcmarkup_category", None)
            context.user_data.pop("svcmarkup_categories", None)
            context.user_data.pop("svcmarkup_subtype", None)
            context.user_data.pop("svcmarkup_subtypes", None)

            kb = InlineKeyboardMarkup(footer_admin())
            await update.message.reply_text(
                f"✅ Updated {updated} services in {plat} - {cat} to {pct}%.",
                reply_markup=kb,
            )
            return

        service_id = int(context.user_data.get("svcmarkup_service_id") or 0)
        if service_id <= 0:
            context.user_data["input_mode"] = INP_NONE
            await update.message.reply_text("Missing service. Start again: Admin → Set Service Markup.")
            return

        Session = context.application.bot_data["Session"]
        async with Session() as session:
            svc = (await session.execute(select(Service).where(Service.id == service_id))).scalars().first()
            if not svc:
                context.user_data["input_mode"] = INP_NONE
                await update.message.reply_text("Service not found. Start again: Admin → Set Service Markup.")
                return
            svc.custom_markup = float(pct)
            await commit_with_retry(session)

        context.user_data["input_mode"] = INP_NONE
        context.user_data.pop("svcmarkup_service_id", None)
        kb = InlineKeyboardMarkup(footer_admin())
        await update.message.reply_text(
            f"✅ Updated! {svc.name} markup is now {pct}%.",
            reply_markup=kb,
        )
        return

    context.user_data["input_mode"] = INP_NONE

async def order_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _maintenance_blocker(update, context):
        return

    q = update.callback_query
    await _safe_answer(q)
    data = q.data

    if data == "order:cancel":
        for k in ("order_service_id", "order_vendor_name", "order_price_display", "order_link", "order_qty"):
            context.user_data.pop(k, None)
        context.user_data["input_mode"] = INP_NONE
        platform = context.user_data.get("sel_platform")
        if platform:
            await screen_platform_categories(update, context, str(platform))
        else:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
            await send_or_edit(update, "❌ Order cancelled.", kb)
        return

    if data == "order:confirm":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])
        await send_or_edit(update, "This step is no longer needed. Please select a service again.", kb)
        return

async def render_state(update: Update, context: ContextTypes.DEFAULT_TYPE, screen: str, payload: Dict[str, Any]) -> None:
    if screen == MAIN:
        await screen_main(update, context)
    elif screen == PLATFORMS:
        await screen_platforms(update, context)
    elif screen == PLAT_CATEGORIES:
        await screen_platform_categories(update, context, payload.get("platform", "Instagram"))
    elif screen == PLAT_SUBTYPES:
        await screen_platform_subtypes(update, context, payload.get("platform", "Instagram"), payload.get("category", ""))
    elif screen == SERVICE_LIST:
        await screen_service_list(update, context, payload.get("platform", "Instagram"), payload.get("category", ""), payload.get("subtype", ""))
    elif screen == WALLET:
        await screen_wallet(update, context)
    elif screen == DEPOSIT:
        await screen_deposit(update, context)
    elif screen == TX:
        await screen_transactions(update, context)
    elif screen == ORDERS:
        await screen_orders(update, context)
    elif screen == ORDER_DETAIL:
        await screen_order_detail(update, context, int(payload.get("order_id", 0)))
    elif screen == SUPPORT:
        await screen_support(update, context)
    elif screen == ADMIN:
        await screen_admin(update, context)
    elif screen == ADMIN_VENDORS:
        await screen_admin_vendors(update, context)
    elif screen == ADMIN_VENDOR_PICK:
        mode = payload.get("mode") or context.user_data.get("vendor_pick_mode") or "test"
        await screen_vendor_list(update, context, mode=str(mode))
    elif screen == ADMIN_VENDOR_STATS:
        await screen_vendor_stats(update, context)
    elif screen == ADMIN_DAILY_STATS:
        await screen_daily_stats(update, context)
    else:
        await screen_main(update, context)


# -----------------------------
# Startup (PostgreSQL bootstrap)
# -----------------------------
async def startup(settings) -> None:
    """Fast startup for PostgreSQL:
      - create tables if missing
      - bootstrap env vendors/settings into DB (no network)

    IMPORTANT: no network calls here.
    """
    # 1) Ensure schema exists
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Backward-compatible schema upgrade: older DBs may not have receipt message columns.
        # create_all() won't add columns to existing tables, so we ensure these exist explicitly.
        try:
            await conn.execute(text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS receipt_message_id BIGINT"))
            await conn.execute(text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS receipt_chat_id BIGINT"))
        except Exception:
            # If the DB user lacks ALTER privileges, the bot can still run; receipt editing just won't persist.
            pass

    # 2) Bootstrap application settings (markup) + vendors
    async with async_session() as session:
        # If vendors table is empty (fresh Postgres DB), seed from .env VENDOR_1..VENDOR_4
        vendor_count = (await session.execute(select(func.count(Vendor.id)))).scalar_one()
        if vendor_count == 0:
            for cfg in settings.vendors:
                try:
                    name = cfg.get("name") if isinstance(cfg, dict) else getattr(cfg, "name", None)
                    url = cfg.get("url") if isinstance(cfg, dict) else getattr(cfg, "url", None)
                    key = cfg.get("key") if isinstance(cfg, dict) else getattr(cfg, "key", None)
                    if not (name and url and key):
                        logger.warning("Skipping invalid vendor config in settings.vendors: %r", cfg)
                        continue
                    session.add(Vendor(name=name, url=url, api_key=key, is_active=True))
                except Exception:
                    logger.exception("Failed to seed vendor from settings.vendors: %r", cfg)
            await commit_with_retry(session)
            logger.info("Bootstrapped %d vendors from .env into PostgreSQL.", len(settings.vendors))

        # Always call bootstrap to ensure markup exists and vendor keys/urls are up to date
        await bootstrap_from_env(settings, session)


async def _initial_sync_task(app: Application) -> None:
    """
    Background task: sync catalog from active vendors (do not block polling).
    """
    try:
        await asyncio.sleep(0.5)
        async with async_session() as session:
            try:
                await asyncio.wait_for(sync_catalog(session), timeout=180)
                logger.info("Initial catalog sync completed.")
            except asyncio.TimeoutError:
                logger.warning("Initial catalog sync timed out (180s). Bot remains usable.")
            except Exception as e:
                logger.warning("Initial catalog sync failed: %s", e)
    except Exception as e:
        logger.exception("Background initial sync task crashed: %s", e)






async def _admin_catalog_sync_task(app: Application, chat_id: int, message_id: int) -> None:
    """Run catalog sync in background and then update the admin message with results."""
    try:
        async with async_session() as session:
            try:
                res = await asyncio.wait_for(sync_catalog(session), timeout=300)
            except asyncio.TimeoutError:
                await app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="⚠️ Catalog Sync timed out (300s). Bot is still running.",
                    reply_markup=InlineKeyboardMarkup(footer_admin()),
                )
                return

        # Keep the completion message short to avoid Telegram message size limits.
        created = int(res.get("created", 0) or 0)
        updated = int(res.get("updated", 0) or 0)
        grouped = int(res.get("services_grouped", 0) or 0)
        vendors = int(res.get("vendors", 0) or 0)
        vendors_ok = int(res.get("vendors_ok", 0) or 0)

        text = (
            "✅ Sync completed. "
            f"{created + updated} services updated.\n\n"
            f"Active vendors: {vendors} (OK: {vendors_ok})\n"
            f"Grouped services: {grouped}"
        )

        # Absolute safety cap (Telegram hard limit is ~4096 chars for most text messages)
        if len(text) > 4000:
            text = text[:3990] + "…"

        await app.bot.edit_message_text(

            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(footer_admin()),
        )
    except Exception as e:
        try:
            await app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"❌ Catalog Sync failed: {e}",
                reply_markup=InlineKeyboardMarkup(footer_admin()),
            )
        except Exception:
            pass

async def _order_status_poller(app: Application) -> None:
    """Poll vendor order statuses every 20 seconds and notify users once on completion."""
    await asyncio.sleep(2.0)
    while True:
        try:
            async with async_session() as session:
                # Use an explicit transaction context so that any error automatically
                # rolls back and we don't leave the session in a failed transaction
                # state (prevents InFailedSQLTransactionError on Postgres).
                async with session.begin():
                    # Lightweight heartbeat so you can see the poller is running.
                    try:
                        cnt = (await session.execute(
                            select(func.count()).select_from(Order).where(Order.status.in_(["Pending", "Processing"]))
                        )).scalar_one()
                        if cnt:
                            logger.info("Order poller tick: checking %s in-flight orders", cnt)
                    except Exception:
                        pass

                    notes = await poll_processing_orders(session, app, limit=50)
            if notes:
                for n in notes:
                    try:
                        telegram_id = int(n["telegram_id"])
                        order_id = int(n["order_id"])
                        service_name = str(n.get("service_name") or "")
                        qty = int(n.get("quantity") or 0)

                        # Prefer showing the vendor/website order id as the primary reference.
                        website_order_id = None
                        target_link = None
                        status_txt = "Completed"
                        try:
                            async with async_session() as s2:
                                res = await s2.execute(
                                    select(Order.vendor_order_id, Order.target, Order.status).where(Order.id == order_id)
                                )
                                row = res.first()
                                if row:
                                    website_order_id, target_link, status_txt = row
                        except Exception:
                            pass

                        # Use values returned by poll_processing_orders (fixes N/A)
                        website_order_id = n.get("vendor_order_id")
                        target_link = n.get("link")
                        import html as _html
                        website_order_id_str = str(website_order_id) if website_order_id is not None else "N/A"
                        # Prefer editing the original receipt (if we saved its message_id) instead of sending a new completion message.
                        edited_receipt = False
                        try:
                            async with async_session() as s3:
                                o = (await s3.execute(select(Order).where(Order.id == order_id))).scalars().first()
                                if o and o.receipt_message_id and o.receipt_chat_id:
                                    svc = (await s3.execute(select(Service).where(Service.id == o.service_id))).scalars().first()
                                    if svc:
                                        receipt_text = _build_order_receipt_text(
                                            svc,
                                            float(getattr(o, "charge", 0.0) or 0.0),
                                            int(getattr(o, "quantity", 0) or 0),
                                            "Completed",
                                        )
                                        try:
                                            await app.bot.edit_message_text(
                                                chat_id=int(o.receipt_chat_id),
                                                message_id=int(o.receipt_message_id),
                                                text=receipt_text,
                                                disable_web_page_preview=True,
                                            )
                                            edited_receipt = True
                                        except (telegram.error.BadRequest, telegram.error.Forbidden, telegram.error.TimedOut):
                                            edited_receipt = False
                        except Exception:
                            edited_receipt = False
                        msg = (
                            "🎉✅ <b>Order Completed!</b>\n\n"
                            f"<b>Order ID:</b> <code>{html.escape(website_order_id_str)}</code>\n"
                            f"📦 <b>Service:</b> {html.escape(service_name)}\n"
                            f"🔢 <b>Quantity:</b> {qty}\n"
                            f"🔗 <b>Link:</b> {html.escape(str(target_link) if target_link else 'N/A')}\n"
                            f"✅ <b>Status:</b> {html.escape(str(status_txt) if status_txt else 'Completed')}"
                        )
                        await app.bot.send_message(
                            chat_id=telegram_id,
                            text=msg,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except Exception as e:
                        logger.exception(f"Error in poller: {e}")
                        continue
        except Exception as e:
            logger.warning("Order status poller error: %s", e)
        await asyncio.sleep(20)

async def _post_init(app: Application) -> None:
    settings = app.bot_data["settings"]

    await startup(settings)

    try:
        async with async_session() as session:
            app.bot_data["markup_percent"] = await get_markup_percent(session)
    except Exception as e:
        logger.warning("Could not load markup percent: %s", e)
        app.bot_data["markup_percent"] = float(getattr(settings, "default_markup_percent", 10.0) or 10.0)

    try:
        app.create_task(_initial_sync_task(app))
    except Exception:
        asyncio.create_task(_initial_sync_task(app))

    # background order status poller (20s)
    try:
        app.create_task(_order_status_poller(app))
    except Exception:
        asyncio.create_task(_order_status_poller(app))

    # background Crypto Pay deposit poller (20s)
    try:
        app.create_task(_crypto_pay_poller(app))
    except Exception:
        asyncio.create_task(_crypto_pay_poller(app))



async def _crypto_pay_poller(app: Application) -> None:
    """Background poller to credit Crypto Pay deposits automatically."""
    settings = app.bot_data.get("settings")
    Session = app.bot_data.get("Session")
    token = getattr(settings, "cryptopay_token", None) if settings else None
    if not token or not Session:
        logger.info("Crypto Pay poller disabled (missing token or Session).")
        return

    crypto = CryptoPayClient(api_token=str(token))

    while True:
        try:
            async with Session() as session:
                credited = await check_crypto_deposits(session, crypto, limit=200)
                if credited:
                    logger.info("Crypto Pay poller: credited %s transaction(s).", credited)
        except Exception as e:
            logger.exception("Crypto Pay poller error: %s", e)

        await asyncio.sleep(20)



def _register_handlers(app: Application) -> None:
    # /start is still supported for onboarding; UI remains inline-only afterwards.
    app.add_handler(CommandHandler("start", start_cmd))

    # Generic nav
    app.add_handler(CallbackQueryHandler(nav_cb, pattern=r"^nav:"))

    # Linear Order Conversation (Service -> Link -> Qty -> Confirm)
    order_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(order_conv_entry_from_service, pattern=r"^svc:")],
        states={
            ORDER_CONV_LINK: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, order_conv_link_input),
                # If the user taps another package button while we are waiting for a link,
                # let them re-select a service instead of silently ignoring the click.
                CallbackQueryHandler(order_conv_entry_from_service, pattern=r"^svc:"),
            ],
            ORDER_CONV_QTY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, order_conv_qty_input),
                # If the user taps another package button while we are waiting for quantity,
                # let them re-select a service instead of silently ignoring the click.
                CallbackQueryHandler(order_conv_entry_from_service, pattern=r"^svc:"),
            ],
            # Allow re-selecting a service while still inside the conversation (used by Cancel->back to list UX)
            ORDER_CONV_CONFIRM: [
                CallbackQueryHandler(order_conv_confirm_cb, pattern=r"^ordcfm:"),
                CallbackQueryHandler(order_conv_entry_from_service, pattern=r"^svc:"),
            ],
        },
        fallbacks=[CallbackQueryHandler(order_conv_fallback_cancel, pattern=r"^ordcfm:cancel$")],
        # IMPORTANT: Without allow_reentry, an active conversation can "swallow" svc:* clicks
        # (e.g., user scrolls up and taps another package) and the bot appears unresponsive.
        allow_reentry=True,
        name="order_conv",
        persistent=False,
    )
    app.add_handler(order_conv)

    # Platform flow: platform -> category -> subtype -> services -> input
    app.add_handler(CallbackQueryHandler(plat_cb, pattern=r"^plat:"))
    app.add_handler(CallbackQueryHandler(cat_cb, pattern=r"^cat:"))
    app.add_handler(CallbackQueryHandler(sub_cb, pattern=r"^sub:"))
    app.add_handler(CallbackQueryHandler(svcpage_cb, pattern=r"^svcpage:"))
    app.add_handler(CallbackQueryHandler(ordpage_cb, pattern=r"^ordpage:"))
    app.add_handler(CallbackQueryHandler(txpage_cb, pattern=r"^txpage:"))

    # Order confirmation (before link/qty)
    app.add_handler(CallbackQueryHandler(oc_cb, pattern=r"^oc:"))

    # Orders / wallet / payments
    app.add_handler(CallbackQueryHandler(ord_cb, pattern=r"^ord:"))
    app.add_handler(CallbackQueryHandler(dep_cb, pattern=r"^dep:"))
    app.add_handler(CallbackQueryHandler(pay_cb, pattern=r"^pay:"))
    app.add_handler(CallbackQueryHandler(order_cb, pattern=r"^order:"))

    # Admin
    app.add_handler(CallbackQueryHandler(mm_cb, pattern=r"^mm:"))
    app.add_handler(CallbackQueryHandler(adm_cb, pattern=r"^adm:"))
    app.add_handler(CallbackQueryHandler(svcmarkup_cb, pattern=r"^svcmarkup:"))
    app.add_handler(CallbackQueryHandler(vend_cb, pattern=r"^vend:"))

    # Text input router (only when prompted)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))


def main() -> None:
    settings = load_settings()

    # Ensure DB engine/session are initialized without requiring BOT_TOKEN for database imports
    global engine, async_session
    engine, async_session = init_engine(getattr(settings, "database_url", None))

    app = Application.builder().token(settings.bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["engine"] = engine
    app.bot_data["Session"] = async_session
    app.bot_data["markup_percent"] = float(getattr(settings, "default_markup_percent", 10.0) or 10.0)

    _register_handlers(app)

    # run async startup inside PTB lifecycle (fast)
    app.post_init = _post_init

    # Standard v20+ start (synchronous).
    # Do NOT wrap main() in asyncio.run() (PTB owns the event loop).
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
