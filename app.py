# app.py
import os
import re
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ---------- Optional finance data (price/market intents) ----------
try:
    import yfinance as yf
except Exception:
    yf = None

load_dotenv(override=True)

# ---------- Local modules ----------
from db import (
    init_db, upsert_articles, save_sentiments,
    fetch_article_history, fetch_sentiment_trend,  # harmless import
)
engine = init_db()

from news_api import fetch_news
from text_utils import build_document_text, simple_chunk
from embed_store import get_vector_store
from summarizer_local import summarize_chunks_local
from sentiment import finbert_sentiment_batch
from rag_llm import rag_answer, rag_answer_extractive, _build_prompt

# ====================== Page & Styles ======================
st.set_page_config(page_title="StockInsights", page_icon="📊", layout="wide")
st.title("📊 StockInsights")

st.markdown(
    """
    <style>
      .card-wrap { padding: 0.9rem 1rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #fff; }
      .card-wrap h4 { margin: 0 0 0.5rem 0; }
      .chip { display: inline-block; padding: 2px 8px; border: 1px solid #e5e7eb; border-radius: 999px; font-size: 12px; margin-right: 6px; color: #374151; background:#fafafa;}
      .muted { color: #6b7280; font-size: 12px; }
      .answer-title { margin: 0 0 0.4rem 0; font-weight: 600; }
      .tight ul { margin-top: 0.2rem; }
      .tight li { margin: 0.2rem 0; }
      /* prevent any article html from messing layout */
      .stMarkdown p, .stMarkdown ul, .stMarkdown li { overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- DataFrame shim (Streamlit Cloud 1.37.x) ----
def show_df(df: pd.DataFrame, height: int = 220):
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height)

# ====================== Sidebar ======================
st.sidebar.header("⚙️ Settings")

# Quick picks
if "company_input" not in st.session_state:
    st.session_state.company_input = os.getenv("DEFAULT_TICKER", "Tesla")

quick = st.sidebar.selectbox(
    "Quick ticker",
    options=["(custom)", "TSLA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
    index=0,
)
if quick != "(custom)":
    st.session_state.company_input = quick

company = st.sidebar.text_input("Company / Ticker", value=st.session_state.company_input, key="company_input")

# News knobs
st.sidebar.subheader("News fetch")
days = st.sidebar.slider("Lookback (days)", 1, 30, 7)
page_size = st.sidebar.slider("Articles per fetch", 10, 100, 50, step=10)
strict_in_title = st.sidebar.checkbox("Strict: require company in title (qInTitle)", value=True)
post_filter_anywhere = st.sidebar.checkbox("Local filter: company must appear in title/desc/content", value=True)
language = st.sidebar.selectbox("Language", ["en"], index=0)
sort_by = st.sidebar.selectbox("Sort by", ["publishedAt", "relevancy", "popularity"], index=0)
inc_sources = st.sidebar.text_input("Include sources (CSV IDs)", placeholder="bloomberg,the-verge")
exc_domains = st.sidebar.text_input("Exclude domains (CSV)", placeholder="example.com,foo.org")

st.sidebar.subheader("Index / Retrieve")
chunk_size = st.sidebar.slider("Chunk size (characters)", 400, 1500, 900, step=100)
k_retrieve = st.sidebar.slider("Top-k for Retrieve", 3, 15, 5)
k_summary = st.sidebar.slider("Top-k for Summary", 3, 15, 8)
focus_retrieval_with_company = st.sidebar.checkbox(
    "Always include company in queries", value=True,
    help="Keeps retrieval/RAG on-topic.",
)

st.sidebar.markdown("---")

# Vector backend
vector_backend = st.sidebar.selectbox(
    "Vector backend",
    options=["faiss", "chroma", "weaviate"],
    index=["faiss", "chroma", "weaviate"].index((os.getenv("VECTOR_BACKEND") or "faiss").lower()),
)
st.sidebar.caption("Set VECTOR_BACKEND env var to persist across runs.")
if vector_backend == "chroma":
    st.sidebar.caption(f"CHROMA_PATH={os.getenv('CHROMA_PATH') or '(memory)'}")
if vector_backend == "weaviate":
    st.sidebar.caption("Requires WEAVIATE_URL (+ optional WEAVIATE_API_KEY, WEAVIATE_CLASS).")

# Generation toggle
use_rag_generation = st.sidebar.toggle(
    "Use LLM generation (RAG)",
    value=False,
    help="OFF = extractive bullets (no LLM). ON = generate via configured backend (e.g., Ollama).",
)

# Model selection (LLM only)
RECOMMENDED_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "google/flan-t5-base",
]
env_default = (os.getenv("HF_MODEL_ID") or "HuggingFaceH4/zephyr-7b-beta").strip()
options = list(dict.fromkeys([env_default] + RECOMMENDED_MODELS))

use_env_model = st.sidebar.toggle("Use model from .env (HF_MODEL_ID)", value=True)
hf_model = st.sidebar.selectbox("Hugging Face model (used only if generation is ON)", options=options)
st.sidebar.caption(f"Current .env HF_MODEL_ID = `{env_default}`")
if any(bad in hf_model.lower() for bad in ["gpt2", "llama-2-7b", "base-only"]):
    st.sidebar.error("This model may not be instruction-tuned. Prefer Zephyr/Mistral/Qwen/Gemma-IT/Flan-T5.")

# ====================== Session ======================
if "articles" not in st.session_state:
    st.session_state.articles = []
if "vs_backend" not in st.session_state:
    st.session_state.vs_backend = None
if "vs" not in st.session_state:
    st.session_state.vs = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ====================== Helpers ======================
@st.cache_data(show_spinner=False, ttl=600)
def cached_fetch_news(q: str, days: int, page_size: int,
                      strict_in_title: bool, language: str, sort_by: str,
                      include_sources: tuple, exclude_domains: tuple):
    return fetch_news(
        q, days=days, page_size=page_size,
        strict_in_title=strict_in_title,
        language=language, sort_by=sort_by,
        include_sources=include_sources, exclude_domains=exclude_domains,
    )

@st.cache_resource(show_spinner=False)
def cached_vector_store(chunks, backend: str):
    vs = get_vector_store(backend)
    vs.build(chunks)
    return vs

def _csv_to_tuple(s: str):
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    return tuple(items)

# ---------- Time & finance helpers ----------
def _infer_ticker(raw: str) -> str:
    if not raw: return ""
    raw = raw.strip()
    if re.fullmatch(r"[A-Z\.]{1,6}", raw): return raw.upper()
    MAP = {"tesla":"TSLA","apple":"AAPL","microsoft":"MSFT","amazon":"AMZN","alphabet":"GOOGL","google":"GOOGL","meta":"META","nvidia":"NVDA"}
    return MAP.get(raw.lower(), raw.upper())

def _yahoo_history_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker}/history"

def _extract_window(q: str):
    """Return (start_date, end_date, label) as dates. Supports: last week/month, past N days, yesterday, today (default: past 7 days)."""
    today = datetime.now(timezone.utc).astimezone().date()
    ql = (q or "").lower()

    if "today" in ql:
        return today, today, "today"
    if "yesterday" in ql:
        y = today - timedelta(days=1); return y, y, "yesterday"
    if "last week" in ql or "past week" in ql:
        monday_this = today - timedelta(days=today.weekday())
        start = monday_this - timedelta(days=7)
        end = monday_this - timedelta(days=1)
        return start, end, "last week"
    if "last month" in ql or "past month" in ql:
        first_this = today.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
        return start, end, "last month"
    m = re.search(r"(last|past)\s+(\d+)\s+days?", ql)
    if m:
        d = int(m.group(2)); return today - timedelta(days=d), today, f"past {d} days"
    return today - timedelta(days=7), today, "past 7 days"

def _looks_like_finance_q(q: str) -> bool:
    ql = (q or "").lower()
    return bool(
        re.search(r"\b(avg|average|mean)\s+(price|close)\b", ql) or
        re.search(r"\b(last|past)\s+(week|month|(\d+)\s+days?)\b.*\b(price|close)\b", ql) or
        re.search(r"\b(price|close)\b.*\b(last|past)\s+(week|month|(\d+)\s+days?)\b", ql) or
        ("yesterday" in ql and "price" in ql)
    )

def _fetch_avg_price(ticker: str, start_d, end_d):
    """Mean Close over [start_d, end_d] inclusive using yfinance; robust to column variants."""
    if yf is None:
        raise RuntimeError("yfinance not installed. Run `pip install yfinance`.")
    df = yf.download(
        ticker,
        start=pd.Timestamp(start_d).tz_localize(None),
        end=pd.Timestamp(end_d + timedelta(days=1)).tz_localize(None),
        interval="1d", progress=False, auto_adjust=True, actions=False,
    )
    if df is None or df.empty:
        return None, pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            elif ticker in df.columns.get_level_values(0):
                df = df.xs(ticker, axis=1, level=0)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    rename_map = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=rename_map)
    close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if not close_col:
        for c in df.columns:
            if "close" in c.lower():
                close_col = c; break
    if not close_col:
        return None, df
    df = df.dropna(subset=[close_col])
    if df.empty:
        return None, pd.DataFrame()
    avg_close = float(df[close_col].mean())
    cols = [close_col] + [c for c in ["Open","High","Low","Volume"] if c in df.columns]
    return avg_close, df[cols]

def _looks_like_market_q(q: str) -> bool:
    ql = (q or "").lower()
    return ("market" in ql) and bool(re.search(r"\b(up|down|today|yesterday|week|month|days?)\b", ql))

_INDEX_MAP = {
    "^GSPC": ("S&P 500",  "https://finance.yahoo.com/quote/%5EGSPC"),
    "^IXIC": ("Nasdaq",   "https://finance.yahoo.com/quote/%5EIXIC"),
    "^DJI":  ("Dow Jones","https://finance.yahoo.com/quote/%5EDJI"),
}

def _fetch_index_moves(symbols, start_d, end_d):
    if yf is None:
        raise RuntimeError("yfinance not installed. Run `pip install yfinance`.")
    out = []
    for sym in symbols:
        df = yf.download(
            sym,
            start=pd.Timestamp(start_d).tz_localize(None),
            end=pd.Timestamp(end_d + timedelta(days=1)).tz_localize(None),
            interval="1d", progress=False, auto_adjust=True,
        )
        if df is None or df.empty or df["Close"].dropna().empty:
            continue
        s = df["Close"].iloc[0].item()
        e = df["Close"].iloc[-1].item()

        pct = (e - s) / s * 100.0
        out.append((sym, s, e, pct))
    return out

# ---------- Timeline intent ("what happened last week?") ----------
def _looks_like_timeline_q(q: str) -> bool:
    ql = (q or "").lower()
    return ("what happened" in ql) or ("highlights" in ql) or ("summary" in ql)

def _summarize_window_from_articles(articles, start_d, end_d, max_articles=12, bullets=8):
    """Filter loaded articles by date window and summarize with local summarizer."""
    # Parse and filter by publishedAt
    def _to_date(s):
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None

    subset = []
    for a in articles:
        d = _to_date(a.get("publishedAt"))
        if d and (start_d <= d <= end_d):
            subset.append(a)

    # Take up to max_articles, newest first
    subset.sort(key=lambda x: x.get("publishedAt") or "", reverse=True)
    subset = subset[:max_articles]

    texts = []
    for a in subset:
        doc = build_document_text(a)
        if doc:
            texts.append(doc)

    if not texts:
        return None, []

    summary = summarize_chunks_local(texts, bullets=bullets)

    # Build sources list
    sources = []
    for a in subset:
        sources.append((
            a.get("title") or "(no title)",
            a.get("url"),
            a.get("source") or "?",
            a.get("publishedAt") or "?",
        ))
    return summary, sources

# ---------- Generic QA helper (maps [i] -> links) ----------
def _answer_with_sources(query: str, vector_store, k: int = 5, num_bullets: int = 6):
    hits = vector_store.search(query, k=k)
    if not hits:
        return "I couldn't find relevant context in the knowledge base.", [], []

    index_meta = {i + 1: meta for i, (meta, _score) in enumerate(hits)}
    answer = rag_answer_extractive(query, vector_store=vector_store, k=k, num_bullets=num_bullets)

    cited, seen = [], set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        idx = int(m.group(1))
        if idx in index_meta and idx not in seen:
            meta = index_meta[idx]
            cited.append((
                idx,
                meta.get("title", "(no title)"),
                meta.get("url"),
                meta.get("source", "?"),
                meta.get("publishedAt", "?"),
            ))
            seen.add(idx)

    top_matches = []
    for i, (meta, score) in enumerate(hits, start=1):
        top_matches.append((
            i, meta.get("title","(no title)"), meta.get("url"),
            meta.get("source","?"), meta.get("publishedAt","?"), score
        ))

    return answer, cited, top_matches

# ====================== Fetch News ======================
if st.button("Fetch News"):
    if not company.strip():
        st.warning("Please enter a company/ticker.")
        st.stop()

    include_sources = _csv_to_tuple(inc_sources)
    exclude_domains = _csv_to_tuple(exc_domains)

    with st.spinner("Fetching news..."):
        try:
            articles = cached_fetch_news(
                company.strip(), days=days, page_size=page_size,
                strict_in_title=strict_in_title, language=language, sort_by=sort_by,
                include_sources=include_sources, exclude_domains=exclude_domains,
            )
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            articles = []

    if post_filter_anywhere and articles:
        key = company.strip().lower()
        def _text(a): return f"{a.get('title','')} {a.get('description','')} {a.get('content','')}".lower()
        articles = [a for a in articles if key and key in _text(a)]

    st.session_state.articles = articles
    st.subheader(f"Found {len(articles)} articles for ‘{company}’")

    try:
        inserted = upsert_articles(engine, company.strip(), articles)
        st.caption(f"Saved {inserted} article records to the database (deduped by company+url).")
    except Exception as e:
        st.caption(f"DB save (articles) skipped due to: {e}")

    # Preview
    for i, a in enumerate(articles, 1):
        title = a.get("title") or "(no title)"
        with st.expander(f"{i}. {title} — {a.get('source')} ({a.get('publishedAt')})"):
            if a.get("url"): st.markdown(f"[Open article]({a['url']})")
            excerpt = (build_document_text(a) or "")[:600]
            st.write((excerpt + ("..." if len(excerpt) == 600 else "")) or "No excerpt available.")

    # Build chunks for vector indexing
    chunks = []
    for a in articles:
        doc = build_document_text(a)
        if not doc: continue
        for piece in simple_chunk(doc, max_chars=chunk_size):
            chunks.append((piece, {
                "text": piece, "title": a.get("title"), "url": a.get("url"),
                "source": a.get("source"), "publishedAt": a.get("publishedAt"),
            }))
    st.session_state.chunks = chunks

    # Build vector store
    try:
        st.session_state.vs = cached_vector_store(chunks, backend=vector_backend)
        st.session_state.vs_backend = vector_backend
        st.success(f"Indexed {len(chunks)} chunks with backend = {vector_backend} (chunk_size={chunk_size}).")
    except Exception as e:
        st.warning(f"Vector backend '{vector_backend}' unavailable ({e}). Falling back to FAISS.")
        try:
            st.session_state.vs = cached_vector_store(chunks, backend="faiss")
            st.session_state.vs_backend = "faiss"
            st.success(f"Indexed {len(chunks)} chunks with FAISS.")
        except Exception as e2:
            st.session_state.vs = None
            st.error(f"Failed to index chunks even with FAISS: {e2}")

# ====================== Retrieval / Summarization / RAG ======================
if st.session_state.vs is not None:
    query = st.text_input("Ask about these articles", key="qry")
    debug = st.checkbox("🔍 Debug mode", value=False)

    col1, col2, col3 = st.columns(3)

    # ---------- Retrieve: Router -> market / price / timeline / generic ----------
    with col1:
        if st.button("Retrieve"):
            q = (query or "summary").strip()
            if focus_retrieval_with_company:
                q = f"{company} {q}".strip()

            # MARKET intent
            if _looks_like_market_q(q):
                try:
                    start_d, end_d, label = _extract_window(q)
                    with st.spinner(f"Checking major indexes for {label}..."):
                        moves = _fetch_index_moves(["^GSPC","^IXIC","^DJI"], start_d, end_d)
                    st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
                    st.markdown('<div class="answer-title">Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="chip">{label}</span>', unsafe_allow_html=True)
                    if not moves:
                        st.info("Couldn't retrieve index data.")
                    else:
                        lines = []
                        for sym, s, e, pct in moves:
                            name, url = _INDEX_MAP.get(sym, (sym, f"https://finance.yahoo.com/quote/{sym}"))
                            arrow = "▲" if pct >= 0 else "▼"
                            lines.append(f"- **{name}**: {arrow} **{pct:+.2f}%** (from {s:,.2f} to {e:,.2f}) — [Source: Yahoo Finance]({url}/history)")
                        st.markdown("\n".join(lines))
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Market lookup failed: {e}")

            # PRICE intent
            elif _looks_like_finance_q(q):
                try:
                    ticker = _infer_ticker(company)
                    if not ticker:
                        st.error("Couldn't infer a ticker from the Company/Ticker field.")
                    else:
                        start_d, end_d, label = _extract_window(q)
                        with st.spinner(f"Fetching {ticker} prices for {label}..."):
                            avg_close, df_used = _fetch_avg_price(ticker, start_d, end_d)
                        st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
                        st.markdown('<div class="answer-title">Answer</div>', unsafe_allow_html=True)
                        st.markdown(f'<span class="chip">{ticker}</span> <span class="chip">{label}</span>', unsafe_allow_html=True)
                        if avg_close is None:
                            st.info(f"No usable close column found for {ticker} in {label}.")
                            if isinstance(df_used, pd.DataFrame) and not df_used.empty:
                                st.caption(f"Available columns: {list(df_used.columns)}")
                                show_df(df_used.reset_index(), height=220)
                        else:
                            st.markdown(
                                f"- **Average close**: **${avg_close:,.2f}** "
                                f"[Source: Yahoo Finance]({_yahoo_history_url(ticker)})"
                            )
                            if not df_used.empty:
                                show_df(df_used.reset_index(), height=220)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Finance lookup failed: {e}")

            # TIMELINE intent ("what happened last week?")
            elif _looks_like_timeline_q(q):
                start_d, end_d, label = _extract_window(q)
                with st.spinner(f"Summarizing articles for {label}..."):
                    summary, sources = _summarize_window_from_articles(st.session_state.get("articles", []), start_d, end_d, bullets=8)
                st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
                st.markdown('<div class="answer-title">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="chip">{label}</span>', unsafe_allow_html=True)
                if not summary:
                    st.info("No articles in that window to summarize.")
                else:
                    # tidy bullets
                    lines = [ln.strip().lstrip("-• ") for ln in summary.splitlines() if ln.strip()]
                    st.markdown("\n".join(f"- {ln}" for ln in lines))
                    if sources:
                        st.markdown("**Sources**")
                        for title, url, src, date in sources:
                            link = f"[{src} — {title}]({url})" if url else f"{src} — {title}"
                            st.markdown(f"- {link} ({date})")
                st.markdown('</div>', unsafe_allow_html=True)

            # GENERIC QA (extractive + sources)
            else:
                with st.spinner("Retrieving & composing answer..."):
                    answer_md, cited, top_matches = _answer_with_sources(q, st.session_state.vs, k=k_retrieve, num_bullets=6)
                st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
                st.markdown('<div class="answer-title">Answer</div>', unsafe_allow_html=True)
                st.markdown(answer_md)
                if cited:
                    st.markdown("**Sources**")
                    for idx, title, url, src, date in cited:
                        link = f"[{src} — {title}]({url})" if url else f"{src} — {title}"
                        st.markdown(f"- [{idx}] {link} ({date})")
                elif top_matches:
                    st.markdown("**Top matches**")
                    for idx, title, url, src, date, score in top_matches:
                        link = f"[{src} — {title}]({url})" if url else f"{src} — {title}"
                        st.markdown(f"- [{idx}] {link} — score `{score:.3f}` ({date})")
                st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Summarize Top Hits (local) ----------
    with col2:
        if st.button("Summarize Top Hits (Local)"):
            user_q = (st.session_state.get("qry") or "").strip() or "summary"
            if focus_retrieval_with_company:
                user_q = f"{company} {user_q}".strip()
            with st.spinner("Retrieving & summarizing..."):
                hits = st.session_state.vs.search(user_q, k=k_summary)
                top_texts = [meta.get("text") for meta, _ in hits if meta.get("text")]
            st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
            st.markdown('<div class="answer-title">Summary (Local)</div>', unsafe_allow_html=True)
            if not top_texts:
                st.info("No text found.")
            else:
                summary = summarize_chunks_local(top_texts, bullets=8)
                lines = [ln.strip().lstrip("-• ") for ln in summary.splitlines() if ln.strip()]
                st.markdown("\n".join(f"- {ln}" for ln in lines))
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Answer (RAG) with optional LLM ----------
    with col3:
        if st.button("Answer (RAG)"):
            user_q = (st.session_state.get("qry") or "").strip()
            if focus_retrieval_with_company:
                user_q = f"{company} {user_q}".strip() or f"What happened about {company}?"
            else:
                user_q = user_q or f"What happened about {company}?"

            if not use_rag_generation:
                with st.spinner("Answering (extractive, no LLM)..."):
                    answer = rag_answer_extractive(user_q, vector_store=st.session_state.vs, k=k_retrieve, num_bullets=6)
                st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
                st.markdown('<div class="answer-title">Answer (Extractive)</div>', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                hits = st.session_state.vs.search(user_q, k=k_retrieve)
                contexts = [meta.get("text") or "" for meta, _ in hits]
                if debug:
                    prompt_preview = _build_prompt(user_q, contexts)
                    st.code(prompt_preview[:1500])
                model_to_use = env_default if use_env_model else hf_model
                with st.spinner(f"Generating with {model_to_use}…"):
                    answer = rag_answer(
                        user_q,
                        vector_store=st.session_state.vs,
                        k=k_retrieve,
                        prefer_model=None if use_env_model else hf_model,
                    )
                st.markdown('<div class="card-wrap tight">', unsafe_allow_html=True)
                st.markdown('<div class="answer-title">LLM Answer (RAG)</div>', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.caption("Fetch news first to build the index.")

# ====================== Sentiment ======================
st.markdown("---")
st.subheader("🧠 Sentiment Analysis (FinBERT)")

def _fmt_pct(p: float) -> str:
    return f"{p:.0f}%" if p >= 5 else ""

def plot_donut_pie(counts: pd.Series, title: str = "Batch Sentiment Share"):
    ordered = counts.reindex(["positive","neutral","negative"]).fillna(0).astype(int)
    values, labels = ordered.values, ordered.index.tolist()
    if values.sum() == 0:
        st.info("No sentiments to chart yet."); return
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(values, labels=labels, autopct=_fmt_pct, startangle=90, pctdistance=0.78)
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle); ax.set_title(title, fontsize=11, pad=8); ax.axis("equal")
    st.pyplot(fig, clear_figure=True)

def plot_bar_counts(counts: pd.Series, title: str = "Batch Sentiment Count"):
    ordered = counts.reindex(["positive","neutral","negative"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(ordered.index, ordered.values); ax.set_ylabel("Articles")
    ax.set_title(title, fontsize=11, pad=8); ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    st.pyplot(fig, clear_figure=True)

has_articles = bool(st.session_state.get("articles"))
analyze_btn = st.button("Analyze Sentiment", disabled=not has_articles)
if not has_articles:
    st.info("Fetch news first, then you can analyze sentiment.")

if analyze_btn and has_articles:
    articles = st.session_state.get("articles", [])
    texts, metas = [], []
    for a in articles:
        doc = build_document_text(a)
        if not doc: continue
        texts.append(doc); metas.append(a)

    if not texts:
        st.info("No article text.")
    else:
        with st.spinner("Classifying sentiment..."):
            labels = finbert_sentiment_batch(texts, batch_size=8)

        rows = []
        for a, lab in zip(metas, labels):
            rows.append({
                "publishedAt": a.get("publishedAt"),
                "source": a.get("source"),
                "title": a.get("title"),
                "url": a.get("url"),
                "sentiment": lab,
            })

        try:
            saved = save_sentiments(engine, company.strip(), rows)
            st.caption(f"Saved {saved} sentiment rows to DB.")
        except Exception as e:
            st.caption(f"Could not save sentiments: {e}")

        df = pd.DataFrame(rows)
        show_df(df, height=220)
        batch_counts = df["sentiment"].value_counts()
        c1, c2 = st.columns(2)
        with c1: plot_donut_pie(batch_counts, title="Batch Sentiment Share")
        with c2: plot_bar_counts(batch_counts, title="Batch Sentiment Count")


##Average price of a stock (numeric + Yahoo link)

##“Is the market up or down?” (S&P 500 / Nasdaq / Dow, % change + links)

##Timeline questions like “what happened last week?” (date-filtered summary + sources)

##Generic questions (extractive QA + linkified sources)