# rag_llm.py — CPU-first RAG: Extractive (no LLM) + optional Ollama
import os
import re
import time
import requests
from typing import List, Dict, Tuple, Optional

# -------------------- Prompt Builder (for Ollama only) --------------------
def _build_prompt(query: str, contexts: List[str], num_bullets: int = 5) -> str:
    joined = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "You are a careful assistant. Answer ONLY using the sources below.\n"
        "If the answer is not contained in the sources, say \"I don't know\".\n"
        f"Respond in {num_bullets} concise bullet points and cite sources inline like [1], [2].\n\n"
        f"Sources:\n{joined}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

# -------------------- Text cleaning --------------------
_WS_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<.*?>")
_JS_NOISE_RE = re.compile(r"{.*?}")  # simple catch for { window.open... }

def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(_HTML_RE, " ", t)          # remove HTML tags
    t = re.sub(_JS_NOISE_RE, " ", t)      # remove inline JS
    t = t.replace("\u00a0", " ")
    t = _WS_RE.sub(" ", t).strip()
    return t.lstrip("-• ").strip()

def _truncate_contexts(contexts: List[str], max_chars_per_ctx: int = 700, max_total_chars: int = 2400) -> List[str]:
    out, total = [], 0
    for c in contexts:
        c = _clean_text(c)[:max_chars_per_ctx]
        if not c:
            continue
        if total + len(c) > max_total_chars:
            break
        out.append(c)
        total += len(c)
    return out

# -------------------- Company tagging --------------------
COMPANIES = ["Tesla", "Apple", "Microsoft", "Google", "Alphabet", "Amazon", "Meta", "Nvidia"]

def _tag_company(sentence: str) -> str:
    for comp in COMPANIES:
        if comp.lower() in sentence.lower():
            return f"[{comp}] {sentence}"
    return sentence

# -------------------- Extractive (CPU-only) --------------------
def rag_answer_extractive(
    query: str,
    vector_store,
    k: int = 5,
    num_bullets: int = 6,
    max_chars_per_ctx: int = 700,
    max_total_ctx_chars: int = 2400,
) -> str:
    """
    Retrieve top-k chunks and output cited bullet points (no generator model).
    """
    results: List[Tuple[Dict, float]] = vector_store.search(query, k=k)
    if not results:
        return "I couldn't find relevant context in the knowledge base."

    raw_contexts = []
    for idx, (meta, _score) in enumerate(results, start=1):
        txt = meta.get("text") or meta.get("chunk") or ""
        if txt:
            raw_contexts.append((idx, _clean_text(txt)))

    kept, total = [], 0
    for sid, c in raw_contexts:
        c = c[:max_chars_per_ctx]
        if not c:
            continue
        if total + len(c) > max_total_ctx_chars:
            break
        kept.append((sid, c))
        total += len(c)

    sent_split = re.compile(r"(?<=[.!?])\s+")
    q_terms = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 2)

    cand = []
    for sid, ctx in kept:
        for s in sent_split.split(ctx):
            s_clean = s.strip()
            if len(s_clean) < 30:
                continue
            tokens = [w.lower() for w in re.findall(r"\w+", s_clean)]
            score = sum(1 for t in tokens if t in q_terms)
            score += min(len(s_clean) // 80, 2)
            cand.append((score, sid, s_clean))

    cand.sort(key=lambda x: x[0], reverse=True)

    bullets, seen = [], set()
    for score, sid, s in cand:
        s = _tag_company(s)  # add company tag if matched
        pref = s[:50].lower()
        if any(pref.startswith(p) or p.startswith(pref) for p in seen):
            continue
        bullets.append(f"- {s} [{sid}]")
        seen.add(pref)
        if len(bullets) >= num_bullets:
            break

    if not bullets:
        for sid, ctx in kept:
            first = sent_split.split(ctx)[0].strip()
            if first:
                bullets.append(f"- {_tag_company(first)} [{sid}]")
            if len(bullets) >= num_bullets:
                break

    return "\n".join(bullets) if bullets else "I couldn't form an extractive answer from the retrieved snippets."

# -------------------- Optional: Ollama local generation --------------------
def _call_ollama_chat(
    prompt: str,
    model_id: str,
    timeout: int = 120,
    temperature: float = 0.2,
    max_new_tokens: int = 192,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Only use the provided context. If missing, say you don't know."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (409, 429, 502, 503, 504):
                time.sleep(retry_backoff * attempt)
                continue
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff * attempt)
    raise RuntimeError(f"Ollama error after retries: {last_err}")

# -------------------- Public RAG --------------------
def rag_answer(
    query: str,
    vector_store,
    k: int = 5,
    prefer_model: Optional[str] = None,
    timeout: int = 120,
    num_bullets: int = 5,
    max_chars_per_ctx: int = 700,
    max_total_ctx_chars: int = 2400,
    temperature: float = 0.2,
) -> str:
    results: List[Tuple[Dict, float]] = vector_store.search(query, k=k)
    if not results:
        return "I couldn't find relevant context in the knowledge base."

    raw_contexts: List[str] = []
    for meta, _score in results:
        txt = meta.get("text") or meta.get("chunk") or ""
        if txt:
            raw_contexts.append(txt)

    contexts = _truncate_contexts(
        raw_contexts, max_chars_per_ctx=max_chars_per_ctx, max_total_chars=max_total_ctx_chars
    )
    if not contexts:
        return "I retrieved results but they had no readable text."

    backend = (os.getenv("RAG_BACKEND") or "").strip().lower()
    if backend == "ollama":
        prompt = _build_prompt(query, contexts, num_bullets=num_bullets)
        model_id = (prefer_model or os.getenv("HF_MODEL_ID") or "tinyllama").strip()
        try:
            return _call_ollama_chat(
                prompt,
                model_id=model_id,
                timeout=timeout,
                temperature=float(os.getenv("RAG_TEMPERATURE", str(temperature))),
                max_new_tokens=int(os.getenv("RAG_MAX_NEW_TOKENS", "192")),
            )
        except Exception as e:
            return f"(Generator unavailable: {e})\n\n" + rag_answer_extractive(
                query, vector_store, k=k, num_bullets=num_bullets,
                max_chars_per_ctx=max_chars_per_ctx, max_total_ctx_chars=max_total_ctx_chars
            )

    return rag_answer_extractive(
        query, vector_store, k=k, num_bullets=num_bullets,
        max_chars_per_ctx=max_chars_per_ctx, max_total_ctx_chars=max_total_ctx_chars
    )
