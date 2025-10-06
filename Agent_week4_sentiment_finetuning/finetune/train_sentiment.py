# finetune/train_sentiment.py
import os
import inspect
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

BASE_MODEL = os.getenv("BASE_MODEL", "distilbert-base-uncased")
OUT_DIR = os.getenv("OUT_DIR", "finetune/adapters-distilbert-imdb")

def make_training_args(**overrides):
    base = dict(
        output_dir="finetune/out",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=2,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )
    # Transformers v4 uses `evaluation_strategy`, v5 uses `eval_strategy`
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        base["eval_strategy"] = "epoch"          # v5
    else:
        base["evaluation_strategy"] = "epoch"    # v4
    base.update(overrides)
    return TrainingArguments(**base)

def main():
    # 1) Data
    ds = load_dataset("imdb")
    ds["train"] = ds["train"].shuffle(seed=42).select(range(4000))
    ds["test"]  = ds["test"].shuffle(seed=42).select(range(1000))

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    def enc(ex):
        return tok(ex["text"], truncation=True, padding="max_length", max_length=256)

    enc_ds = ds.map(enc, batched=True).rename_column("label", "labels")
    enc_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 3) Base model + LoRA
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention proj layers
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    # 4) Metrics
    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "acc": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds),
        }

    # 5) Train
    args = make_training_args()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc_ds["train"],
        eval_dataset=enc_ds["test"],
        compute_metrics=metrics,
    )
    trainer.train()
    print(trainer.evaluate())

    # 6) Save adapter + tokenizer
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Saved LoRA adapter to: {OUT_DIR}")

if __name__ == "__main__":
    main()
