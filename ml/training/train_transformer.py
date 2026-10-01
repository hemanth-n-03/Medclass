"""
Transformer Multi-Task Model Training & Evaluation
Architecture:
Input Question
  ↓
BERT Tokenizer (uncased)
  ↓
Transformer Encoder (Contextual representations)
  ↓
Pooled Representation (Mean + CLS pool)
  ↓
Shared Projection (Dense + LayerNorm + Dropout)
  ↓
Multi-Task Classification Heads:
  - Domain Head: Medical vs Non-Medical
  - Category Head: 13 Medical Categories
  - Intent Head: 10 Medical Intents
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models", "transformer")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_BACKBONE = "prajjwal1/bert-tiny"
TOKENIZER_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 48

class TransformerMultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer, cat2idx, int2idx, max_len=MAX_LEN):
        self.texts = df["clean_question"].tolist()
        self.domains = df["is_medical"].astype(int).tolist()
        self.cats = [cat2idx.get(c, 0) for c in df["category"]]
        self.intents = [int2idx.get(i, 0) for i in df["intent"]]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "domain": torch.tensor(self.domains[idx], dtype=torch.long),
            "category": torch.tensor(self.cats[idx], dtype=torch.long),
            "intent": torch.tensor(self.intents[idx], dtype=torch.long),
        }

class MultiTaskTransformer(nn.Module):
    def __init__(self, backbone_name, num_cats, num_intents, dropout=0.2):
        super().__init__()
        self.transformer = BertModel.from_pretrained(backbone_name)
        hidden_size = self.transformer.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.domain_head = nn.Linear(256, 2)
        self.cat_head = nn.Linear(256, num_cats)
        self.intent_head = nn.Linear(256, num_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Sequence output: [batch, seq_len, hidden]
        seq_out = outputs.last_hidden_state
        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(seq_out.size()).float()
        sum_embeddings = torch.sum(seq_out * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        shared = self.shared_proj(pooled)
        d_logits = self.domain_head(shared)
        c_logits = self.cat_head(shared)
        i_logits = self.intent_head(shared)
        return d_logits, c_logits, i_logits

def evaluate_transformer(model, loader, device, cats, intents):
    model.eval()
    domain_preds, domain_trues = [], []
    cat_preds, cat_trues = [], []
    intent_preds, intent_trues = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dom = batch["domain"]
            cat = batch["category"]
            intent = batch["intent"]

            d_out, c_out, i_out = model(input_ids, attention_mask)

            domain_preds.extend(d_out.argmax(dim=1).cpu().numpy())
            domain_trues.extend(dom.numpy())

            med_mask = (dom == 1).numpy()
            if med_mask.sum() > 0:
                cat_preds.extend(c_out.argmax(dim=1).cpu().numpy()[med_mask])
                cat_trues.extend(cat.numpy()[med_mask])
                intent_preds.extend(i_out.argmax(dim=1).cpu().numpy()[med_mask])
                intent_trues.extend(intent.numpy()[med_mask])

    def calc(y_t, y_p):
        return {
            "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "precision_macro": round(float(precision_score(y_t, y_p, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_t, y_p, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_t, y_p, average="weighted", zero_division=0)), 4),
        }

    return {
        "domain": calc(domain_trues, domain_preds),
        "category": calc(cat_trues, cat_preds),
        "intent": calc(intent_trues, intent_preds),
        "cat_cm": confusion_matrix(cat_trues, cat_preds).tolist(),
        "intent_cm": confusion_matrix(intent_trues, intent_preds).tolist(),
    }

def train_transformer():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    cats = sorted([c for c in train_df["category"].unique() if c != "Non-Medical"])
    intents = sorted([i for i in train_df["intent"].unique() if i != "Non-Medical"])
    cat2idx = {c: i for i, c in enumerate(cats)}
    int2idx = {it: i for i, it in enumerate(intents)}

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    train_ds = TransformerMultiTaskDataset(train_df, tokenizer, cat2idx, int2idx)
    val_ds = TransformerMultiTaskDataset(val_df, tokenizer, cat2idx, int2idx)
    test_ds = TransformerMultiTaskDataset(test_df, tokenizer, cat2idx, int2idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device("cpu")
    model = MultiTaskTransformer(
        backbone_name=MODEL_BACKBONE,
        num_cats=len(cats),
        num_intents=len(intents),
        dropout=0.2
    ).to(device)

    # Optimizer & LR Scheduler
    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    dom_weight = torch.tensor([8.0, 1.0], device=device)
    criterion_dom = nn.CrossEntropyLoss(weight=dom_weight)

    best_val_score = 0.0
    history = []

    print(f"\nStarting Transformer Fine-Tuning ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dom = batch["domain"].to(device)
            cat = batch["category"].to(device)
            intent = batch["intent"].to(device)

            optimizer.zero_grad()
            d_logits, c_logits, i_logits = model(input_ids, attention_mask)

            l_dom = criterion_dom(d_logits, dom)
            med_indices = (dom == 1).nonzero(as_tuple=True)[0]
            if len(med_indices) > 0:
                l_cat = criterion(c_logits[med_indices], cat[med_indices])
                l_int = criterion(i_logits[med_indices], intent[med_indices])
            else:
                l_cat = torch.tensor(0.0, device=device)
                l_int = torch.tensor(0.0, device=device)

            loss = l_dom + 1.2 * l_cat + 1.0 * l_int
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_eval = evaluate_transformer(model, val_loader, device, cats, intents)
        combined_f1 = (val_eval["category"]["macro_f1"] + val_eval["intent"]["macro_f1"]) / 2.0

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_cat_f1": val_eval["category"]["macro_f1"],
            "val_intent_f1": val_eval["intent"]["macro_f1"]
        })
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} | Val Cat F1: {val_eval['category']['macro_f1']:.4f} | Val Intent F1: {val_eval['intent']['macro_f1']:.4f}")

        if combined_f1 > best_val_score:
            best_val_score = combined_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer_multitask.pt"))

    # Load best model checkpoint
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "transformer_multitask.pt")))
    test_results = evaluate_transformer(model, test_loader, device, cats, intents)
    print("\nTransformer Final Test Set Evaluation:")
    print("Domain:", test_results["domain"])
    print("Category:", test_results["category"])
    print("Intent:", test_results["intent"])

    # Save artifacts
    tokenizer.save_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump({"categories": cats, "intents": intents}, f, indent=2)
    with open(os.path.join(MODEL_DIR, "transformer_metrics.json"), "w") as f:
        json.dump({
            "test_metrics": test_results,
            "history": history,
            "backbone": MODEL_BACKBONE,
            "tokenizer": TOKENIZER_NAME,
            "architecture": "BERT Contextual Encoder -> Mean Pooling -> LayerNorm Dense (256) -> Multi-Task Heads"
        }, f, indent=2)

    print(f"Transformer artifacts saved successfully to {MODEL_DIR}")
    return test_results

if __name__ == "__main__":
    train_transformer()
