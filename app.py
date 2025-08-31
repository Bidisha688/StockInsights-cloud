import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# --- Bridge Streamlit Cloud secrets -> env vars (so local .env OR Cloud secrets both work) ---
for key in (
    "NEWSAPI_KEY", "VECTOR_BACKEND", "HF_MODEL_ID", "HF_TOKEN",
    "CHROMA_PATH", "WEAVIATE_URL", "WEAVIATE_API_KEY", "DEFAULT_TICKER",
    "RAG_TEMPERATURE", "RAG_MAX_NEW_TOKENS"
):
    if key in st.secrets:
        os.environ.setdefault(key, str(st.secrets[key]))

load_dotenv(override=True)

from db import (
    init_db, upsert_articles, save_sentiments,
    fetch_article_history, fetch_sentiment_trend,  # harmless import
)
engine = init_db()

from news_api import fetch_news
from text_utils import build_document_text, simple_chunk
from embed_store import get_vector_store  # pluggable backends (faiss/chroma/weaviate with safe fallbacks)
from summarizer_local import summarize_chunks_local
from sentiment import finbert_sentiment_batch
from rag_llm import rag_answer, rag_answer_extractive, _build_prompt

# ------------------- Page setup -------------------
st.set_page_config(page_title="StockInsights", page_icon="📊", layout="wide")
st.title("📊 StockInsights")

# Small CSS helper to make “cards” wrap text nicely (prevents overflow)
st.markdown(
    """
    <style>
      .card-wrap { padding: 0.9rem 1rem; border: 1px solid #ddd; border-radius: 10px; background: #fff; }
      .card-wrap h4 { margin-top: 0; margin-bottom: 0.6rem; }
      .card-wrap p, .card-wrap li { margin: 0.15rem 0; }
      .card-wrap, .card-wrap * { word-wrap: break-word; overflow-wrap: anywhere; white-space: normal; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- DataFrame compatibility shim (Streamlit Cloud is 1.37.x; no width="stretch" yet) ----
def show_df(df: pd.DataFrame, height: int = 220):
    try:
        # Newer Streamlit (>=1.49) supports width="stretch"
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        # Older Streamlit (1.37.x on Cloud) path
        st.dataframe(df, use_container_width=True, height=height)

# ------------------- Sidebar: Settings -------------------
st.sidebar.header("⚙️ Settings")

# Quick-pick ticker plus free text
if "company_input" not in st.session_state:
    st.session_state.company_input = os.getenv("DEFAULT_TICKER", "Tesla")

quick = st.sidebar.selectbox(
    "Quick ticker",
    options=["(custom)", "TSLA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
    index=0,
    help="Choose a popular ticker or keep (custom)."
)
if quick != "(custom)":
    st.session_state.company_input = quick

company = st.sidebar.text_input("Company / Ticker", value=st.session_state.company_input, key="company_input")

# News / indexing knobs
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
    help="Keeps retrieval/RAG on-topic."
)

st.sidebar.markdown("---")

# Vector backend picker
vector_backend = st.sidebar.selectbox(
    "Vector backend",
    options=["faiss", "chroma", "weaviate"],
    index=["faiss", "chroma", "weaviate"].index((os.getenv("VECTOR_BACKEND") or "faiss").lower()),
    help="FAISS = in-process (default). Chroma = local persistent. Weaviate = remote cluster.",
)
st.sidebar.caption("Set VECTOR_BACKEND env var to persist across runs.")
if vector_backend == "chroma":
    st.sidebar.caption(f"CHROMA_PATH={os.getenv('CHROMA_PATH') or '(memory)'}")
if vector_backend == "weaviate":
    st.sidebar.caption("Requires WEAVIATE_URL (+ optional WEAVIATE_API_KEY, WEAVIATE_CLASS).")

# ------------------- Generation toggle -------------------
use_rag_generation = st.sidebar.toggle(
    "Use LLM generation (RAG)",
    value=False,
    help="OFF = extractive bullets (no LLM, CPU-only). ON = generate with your configured backend (e.g., HF/Ollama)."
)

# ------------------- Model selection (only relevant if LLM is ON) -------------------
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

# ------------------- Session state -------------------
if "articles" not in st.session_state:
    st.session_state.articles = []
if "vs_backend" not in st.session_state:
    st.session_state.vs_backend = None
if "vs" not in st.session_state:
    st.session_state.vs = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ------------------- Helpers -------------------
@st.cache_data(show_spinner=False, ttl=600)
def cached_fetch_news(q: str, days: int, page_size: int,
                      strict_in_title: bool, language: str, sort_by: str,
                      include_sources: tuple, exclude_domains: tuple):
    return fetch_news(
        q, days=days, page_size=page_size,
        strict_in_title=strict_in_title,
        language=language, sort_by=sort_by,
        include_sources=include_sources, exclude_domains=exclude_domains
    )

@st.cache_resource(show_spinner=False)
def cached_vector_store(chunks, backend: str):
    vs = get_vector_store(backend)  # returns faiss/chroma/weaviate with safe import fallbacks
    vs.build(chunks)
    return vs

def _csv_to_tuple(s: str):
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    return tuple(items)

# ------------------- Fetch News -------------------
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
                strict_in_title=strict_in_title,
                language=language, sort_by=sort_by,
                include_sources=include_sources, exclude_domains=exclude_domains
            )
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            articles = []

    # Optional local post-filter to keep on-topic even if API returns loose matches
    if post_filter_anywhere and articles:
        key = company.strip().lower()
        def _text(a):
            return f"{a.get('title','')} {a.get('description','')} {a.get('content','')}".lower()
        articles = [a for a in articles if key and key in _text(a)]

    st.session_state.articles = articles
    st.subheader(f"Found {len(articles)} articles for ‘{company}’")

    try:
        inserted = upsert_articles(engine, company.strip(), articles)
        st.caption(f"Saved {inserted} article records to the database (deduped by company+url).")
    except Exception as e:
        st.caption(f"DB save (articles) skipped due to: {e}")

    # Show article previews
    for i, a in enumerate(articles, 1):
        title = a.get("title") or "(no title)"
        with st.expander(f"{i}. {title} — {a.get('source')} ({a.get('publishedAt')})"):
            if a.get("url"):
                st.markdown(f"[Open article]({a['url']})")
            excerpt = (build_document_text(a) or "")[:600]
            st.write((excerpt + ("..." if len(excerpt) == 600 else "")) or "No excerpt available.")

    # Build chunks for vector indexing
    chunks = []
    for a in articles:
        doc = build_document_text(a)
        if not doc:
            continue
        for piece in simple_chunk(doc, max_chars=chunk_size):
            chunks.append((
                piece,
                {
                    "text": piece,
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "source": a.get("source"),
                    "publishedAt": a.get("publishedAt"),
                }
            ))
    st.session_state.chunks = chunks

    # Build vector store with selected backend, falling back if needed
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

# ------------------- Retrieval + Summarization / RAG -------------------
if st.session_state.vs is not None:
    query = st.text_input("Ask about these articles", key="qry")
    debug = st.checkbox("🔍 Debug mode", value=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Retrieve"):
            q = (query or "summary").strip()
            if focus_retrieval_with_company:
                q = f"{company} {q}".strip()
            with st.spinner("Retrieving..."):
                hits = st.session_state.vs.search(q, k=k_retrieve)
            if not hits:
                st.info("No results.")
            else:
                st.markdown("### Top matches")
                for meta, score in hits:
                    st.markdown(
                        f"- **{meta.get('title','(no title)')}** — *{meta.get('source','?')}* "
                        f"({meta.get('publishedAt','?')}) — score `{score:.3f}`"
                    )

    with col2:
        if st.button("Summarize Top Hits (Local)"):
            user_q = (st.session_state.get("qry") or "").strip() or "summary"
            if focus_retrieval_with_company:
                user_q = f"{company} {user_q}".strip()
            with st.spinner("Retrieving & summarizing..."):
                hits = st.session_state.vs.search(user_q, k=k_summary)
                top_texts = [meta.get("text") for meta, _ in hits if meta.get("text")]
                if not top_texts:
                    st.info("No text found.")
                else:
                    summary = summarize_chunks_local(top_texts, bullets=8)

                    # Render summary as a compact card with bullets & wrapping
                    st.markdown('<div class="card-wrap"><h4>Summary (Local)</h4>', unsafe_allow_html=True)
                    lines = [ln.strip().lstrip("-• ") for ln in summary.splitlines() if ln.strip()]
                    bullet_md = "\n".join(f"- {ln}" for ln in lines)
                    st.markdown(bullet_md)
                    st.markdown("</div>", unsafe_allow_html=True)

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
                st.markdown('<div class="card-wrap"><h4>Answer (Extractive, no LLM)</h4>', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)
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
                        prefer_model=None if use_env_model else hf_model
                    )
                if isinstance(answer, str) and answer.startswith("[HF error:"):
                    st.error(answer)
                else:
                    st.markdown('<div class="card-wrap"><h4>LLM Answer (RAG)</h4>', unsafe_allow_html=True)
                    st.markdown(answer)
                    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.caption("Fetch news first to build the index.")

# ------------------- Sentiment Analysis — Pie + Bar only -------------------
st.markdown("---")
st.subheader("🧠 Sentiment Analysis (FinBERT)")

# Small plotting helpers
def _fmt_pct(p: float) -> str:
    return f"{p:.0f}%" if p >= 5 else ""

def plot_donut_pie(counts: pd.Series, title: str = "Batch Sentiment Share"):
    ordered = counts.reindex(["positive", "neutral", "negative"]).fillna(0).astype(int)
    values = ordered.values
    labels = ordered.index.tolist()
    if values.sum() == 0:
        st.info("No sentiments to chart yet.")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(
        values,
        labels=labels,
        autopct=_fmt_pct,
        startangle=90,
        pctdistance=0.78,
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)

def plot_bar_counts(counts: pd.Series, title: str = "Batch Sentiment Count"):
    ordered = counts.reindex(["positive", "neutral", "negative"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(ordered.index, ordered.values)
    ax.set_ylabel("Articles")
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    st.pyplot(fig, clear_figure=True)

# Don't allow running sentiment before fetching news
has_articles = bool(st.session_state.get("articles"))
analyze_btn = st.button("Analyze Sentiment", disabled=not has_articles)
if not has_articles:
    st.info("Fetch news first, then you can analyze sentiment.")

if analyze_btn and has_articles:
    articles = st.session_state.get("articles", [])
    texts, metas = [], []
    for a in articles:
        doc = build_document_text(a)
        if not doc:
            continue
        texts.append(doc)
        metas.append(a)

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
                "sentiment": lab
            })

        # Persist per-article sentiments
        try:
            saved = save_sentiments(engine, company.strip(), rows)
            st.caption(f"Saved {saved} sentiment rows to DB.")
        except Exception as e:
            st.caption(f"Could not save sentiments: {e}")

        df = pd.DataFrame(rows)
        show_df(df, height=220)   # <--- compatibility helper

        batch_counts = df["sentiment"].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            plot_donut_pie(batch_counts, title="Batch Sentiment Share")
        with c2:
            plot_bar_counts(batch_counts, title="Batch Sentiment Count")