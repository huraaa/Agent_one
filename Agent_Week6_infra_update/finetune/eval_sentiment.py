import os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score

BASE_MODEL = os.getenv("BASE_MODEL", "distilbert-base-uncased")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "finetune/adapters-distilbert-imdb")

def prep(ds, tok):
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, padding="max_length", max_length=256),
                batched=True).rename_column("label","labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return ds

def score(model, tok, split):
    logits = []
    labels = []
    for batch in torch.utils.data.DataLoader(split, batch_size=64):
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits.append(out.logits.cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    import numpy as np
    preds = np.argmax(np.vstack(logits), axis=1)
    y = np.concatenate(labels)
    return {"acc": accuracy_score(y, preds), "f1": f1_score(y, preds)}

def main():
    ds = load_dataset("imdb")
    ds["test"] = ds["test"].shuffle(seed=42).select(range(1000))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tst = prep(ds["test"], tok)

    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    base_metrics = score(base, tok, tst)

    lora_base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    lora = PeftModel.from_pretrained(lora_base, ADAPTER_DIR)
    lora_metrics = score(lora, tok, tst)

    print("Base:", base_metrics)
    print("LoRA:", lora_metrics)

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score
    main()
