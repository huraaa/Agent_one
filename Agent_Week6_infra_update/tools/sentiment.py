# tools/sentiment.py
import torch, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = os.getenv("BASE_MODEL", "distilbert-base-uncased")
ADAPTER = os.getenv("ADAPTER_DIR", "finetune/adapters-distilbert-imdb")

_tok = AutoTokenizer.from_pretrained(ADAPTER)
_base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
_model = PeftModel.from_pretrained(_base, ADAPTER).eval()

def sentiment(text: str):
    with torch.no_grad():
        x = _tok(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        logits = _model(**{k:v for k,v in x.items() if k in ("input_ids","attention_mask")}).logits
    idx = int(logits.argmax(dim=1).item())
    prob = float(logits.softmax(dim=1)[0, idx])
    label = "positive" if idx==1 else "negative"
    return {"label": label, "confidence": round(prob, 4)}
