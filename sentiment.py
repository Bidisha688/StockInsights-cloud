# sentiment.py
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL_NAME = "ProsusAI/finbert"

# Load once, reuse across Streamlit reruns
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
_model.eval()
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_DEVICE)

_LABELS = ["negative", "neutral", "positive"]

def finbert_sentiment(text: str) -> str:
    """Classify one text."""
    if not text:
        return "neutral"
    with torch.no_grad():
        inputs = _tokenizer(text[:2000], return_tensors="pt", truncation=True).to(_DEVICE)
        logits = _model(**inputs).logits
        pred = torch.softmax(logits, dim=-1).argmax(-1).item()
    return _LABELS[pred]

def finbert_sentiment_batch(texts: List[str], batch_size: int = 8) -> List[str]:
    """Classify many texts in mini-batches for speed."""
    labels: List[str] = []
    if not texts:
        return labels
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [t[:2000] if t else "" for t in texts[i:i+batch_size]]
            inputs = _tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(_DEVICE)
            logits = _model(**inputs).logits
            preds = torch.softmax(logits, dim=-1).argmax(-1).tolist()
            labels.extend(_LABELS[p] for p in preds)
    return labels
