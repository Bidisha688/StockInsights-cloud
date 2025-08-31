from __future__ import annotations
from typing import Iterable, Optional, List
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (
    create_engine, String, DateTime, Integer, Text, select, func, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

DB_PATH = Path("news_history.db")

# ---------- ORM base ----------
class Base(DeclarativeBase):
    pass

# ---------- Tables ----------
class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company: Mapped[str] = mapped_column(String(64), index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(128), default="")
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(1024), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    __table_args__ = (UniqueConstraint("company", "url", name="uq_company_url"),)


class Sentiment(Base):
    __tablename__ = "sentiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(String(1024), index=True)
    sentiment: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

# ---------- Engine / Init ----------
def get_engine(echo: bool = False):
    return create_engine(f"sqlite:///{DB_PATH.as_posix()}", echo=echo, future=True)

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine

# ---------- Upserts ----------
def upsert_articles(engine, company: str, articles: Iterable[dict]) -> int:
    inserted = 0
    with Session(engine) as s:
        for a in articles:
            try:
                dt = datetime.fromisoformat(
                    (a.get("publishedAt", "") or "").replace("Z", "+00:00")
                )
            except Exception:
                dt = datetime.now(timezone.utc)

            obj = Article(
                company=company,
                published_at=dt,
                source=a.get("source") or "",
                title=a.get("title") or "",
                url=a.get("url") or "",
            )
            try:
                s.add(obj)
                s.commit()
                inserted += 1
            except Exception:
                s.rollback()
    return inserted

def save_sentiments(engine, company: str, rows: Iterable[dict]) -> int:
    with Session(engine) as s:
        saved = 0
        for r in rows:
            try:
                s.add(
                    Sentiment(
                        company=company,
                        url=r.get("url", ""),
                        sentiment=r.get("sentiment", "neutral")
                    )
                )
                s.commit()
                saved += 1
            except Exception:
                s.rollback()
        return saved

# ---------- Queries ----------
def fetch_article_history(engine, company: Optional[str] = None, limit: int = 200) -> List[Article]:
    with Session(engine) as s:
        stmt = select(Article).order_by(Article.published_at.desc()).limit(limit)
        if company:
            stmt = stmt.where(Article.company == company)
        return list(s.execute(stmt).scalars())

def fetch_sentiment_trend(engine, company: Optional[str] = None, days: Optional[int] = None):
    """Return dict[date] -> {positive, neutral, negative} optionally filtered to last N days."""
    with Session(engine) as s:
        stmt = select(
            func.date(Sentiment.created_at).label("d"),
            Sentiment.sentiment,
            func.count().label("c")
        )
        if company:
            stmt = stmt.where(Sentiment.company == company)
        if days and days > 0:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = stmt.where(Sentiment.created_at >= since)

        stmt = stmt.group_by("d", Sentiment.sentiment).order_by("d")
        rows = s.execute(stmt).all()

        trend = {}
        for d, label, c in rows:
            trend.setdefault(d, {"positive": 0, "neutral": 0, "negative": 0})
            trend[d][label] = c
        return trend
