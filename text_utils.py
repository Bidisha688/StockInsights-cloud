import re
from typing import List

def clean_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", s)  # strip HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_document_text(article: dict) -> str:
    parts = [article.get("title",""), article.get("description",""), article.get("content","")]
    return clean_text(" ".join([p for p in parts if p]))

def simple_chunk(text: str, max_chars: int = 1000) -> List[str]:
    text = text.strip()
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
