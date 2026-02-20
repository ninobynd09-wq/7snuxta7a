from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    BigInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(512))


class Vendor(Base):
    __tablename__ = "vendors"
    __table_args__ = (UniqueConstraint("name", name="uq_vendor_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(String(256))
    api_key: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    last_test_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_test_ok: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_ok: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    last_services_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    vendor_services: Mapped[List["VendorService"]] = relationship(back_populates="vendor")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    balance: Mapped[float] = mapped_column(Float, default=0.0)

    orders: Mapped[List["Order"]] = relationship(back_populates="user")
    transactions: Mapped[List["Transaction"]] = relationship(back_populates="user")


class Service(Base):
    __tablename__ = "services"
    __table_args__ = (
        UniqueConstraint("platform", "name", "type", "sub_type", name="uq_service_platform_name_type_subtype"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # UI platform (Instagram, TikTok, ...)
    platform: Mapped[str] = mapped_column(String(32), index=True)

    # Sub-category (Views, Followers, Likes, ...)
    category: Mapped[str] = mapped_column(String(64), index=True)

    # Variant / sub type (e.g., USA - Drop, APV, RAV)
    sub_type: Mapped[str] = mapped_column(String(64), default="", index=True)

    name: Mapped[str] = mapped_column(String(256), index=True)
    type: Mapped[str] = mapped_column(String(128), index=True)

    # cheapest active vendor cost per 1000
    rate: Mapped[float] = mapped_column(Float)
    custom_markup: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min: Mapped[int] = mapped_column(Integer)
    max: Mapped[int] = mapped_column(Integer)

    # Optional receipt specs (may be NULL/empty)
    speed: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    refill: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    start_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    vendor_services: Mapped[List["VendorService"]] = relationship(back_populates="service")
    orders: Mapped[List["Order"]] = relationship(back_populates="service")


class MenuVisibility(Base):
    __tablename__ = "menu_visibility"
    __table_args__ = (
        UniqueConstraint("platform", "category", name="uq_menu_visibility_platform_category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    is_visible: Mapped[bool] = mapped_column(Boolean, default=True, index=True)



class VendorService(Base):
    __tablename__ = "vendor_services"
    __table_args__ = (
        UniqueConstraint("service_id", "vendor_id", name="uq_vendor_service_per_vendor"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service_id: Mapped[int] = mapped_column(ForeignKey("services.id", ondelete="CASCADE"), index=True)
    vendor_id: Mapped[int] = mapped_column(ForeignKey("vendors.id", ondelete="CASCADE"), index=True)

    vendor_service_id: Mapped[str] = mapped_column(String(64), index=True)
    vendor_rate: Mapped[float] = mapped_column(Float)
    vendor_min: Mapped[int] = mapped_column(Integer)
    vendor_max: Mapped[int] = mapped_column(Integer)

    service: Mapped["Service"] = relationship(back_populates="vendor_services")
    vendor: Mapped["Vendor"] = relationship(back_populates="vendor_services")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    service_id: Mapped[int] = mapped_column(ForeignKey("services.id", ondelete="RESTRICT"), index=True)

    link: Mapped[str] = mapped_column(String(1024))
    quantity: Mapped[int] = mapped_column(Integer)

    charge: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    profit: Mapped[float] = mapped_column(Float, default=0.0)

    status: Mapped[str] = mapped_column(String(32), default="Processing")

    vendor_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    vendor_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    receipt_message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    receipt_chat_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship(back_populates="orders")
    service: Mapped["Service"] = relationship(back_populates="orders")




class OrderNotification(Base):
    __tablename__ = 'order_notifications'
    order_id = Column(Integer, primary_key=True)
    completed_notified = Column(Integer, default=0)
    created_at = Column(String)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    amount: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/completed/expired
    payment_payload: Mapped[str] = mapped_column(String(2048))  # JSON string

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship(back_populates="transactions")