"""
Deep Learning Model Training & Evaluation: BiLSTM Multi-Task Architecture
Architecture:
Word Embedding -> Bidirectional LSTM -> Adaptive Pooling -> Dropout ->
Shared Latent Representation ->
  - Domain Classification Head (Medical vs Non-Medical)
  - Category Classification Head (13 clinical specialties)
  - Intent Classification Head (10 clinical intentions)
"""

import os
import json
import re
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models", "bilstm")
os.makedirs(MODEL_DIR, exist_ok=True)

# Tokenizer
def tokenize(text: str):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return text.strip().split()

class Vocab:
    def __init__(self, max_tokens=10000, min_freq=1):
        self.max_tokens = max_tokens
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.w2i = {self.pad_token: 0, self.unk_token: 1}
        self.i2w = {0: self.pad_token, 1: self.unk_token}

    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(tokenize(t))
        for word, count in counts.most_common(self.max_tokens):
            if count >= self.min_freq and word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

    def encode(self, text, max_len=40):
        tokens = tokenize(text)
        ids = [self.w2i.get(w, self.w2i[self.unk_token]) for w in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.w2i[self.pad_token]] * (max_len - len(ids))
        return ids

class MultiTaskDataset(Dataset):
    def __init__(self, df, vocab, cat2idx, int2idx, max_len=40):
        self.data = []
        for _, row in df.iterrows():
            text = row["clean_question"]
            tokens = vocab.encode(text, max_len=max_len)
            domain = int(row["is_medical"])
            cat = cat2idx.get(row["category"], 0)
            intent = int2idx.get(row["intent"], 0)
            self.data.append((
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(domain, dtype=torch.long),
                torch.tensor(cat, dtype=torch.long),
                torch.tensor(intent, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_cats, num_intents, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        rep_dim = hidden_dim * 2
        
        # Shared intermediate projection
        self.shared_dense = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.LayerNorm(rep_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Heads
        self.domain_head = nn.Linear(rep_dim, 2)
        self.cat_head = nn.Linear(rep_dim, num_cats)
        self.intent_head = nn.Linear(rep_dim, num_intents)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        # Global max pooling + mean pooling
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = (mean_pool + max_pool) / 2.0
        
        shared = self.shared_dense(pooled)
        domain_logits = self.domain_head(shared)
        cat_logits = self.cat_head(shared)
        intent_logits = self.intent_head(shared)
        return domain_logits, cat_logits, intent_logits

def evaluate_model(model, loader, device, cats, intents):
    model.eval()
    domain_preds, domain_trues = [], []
    cat_preds, cat_trues = [], []
    intent_preds, intent_trues = [], []

    with torch.no_grad():
        for x, dom, cat, intent in loader:
            x = x.to(device)
            d_out, c_out, i_out = model(x)
            
            domain_preds.extend(d_out.argmax(dim=1).cpu().numpy())
            domain_trues.extend(dom.numpy())

            # Evaluate category and intent only for medical items
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

def train_bilstm():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Categories and Intents
    cats = sorted([c for c in train_df["category"].unique() if c != "Non-Medical"])
    intents = sorted([i for i in train_df["intent"].unique() if i != "Non-Medical"])
    cat2idx = {c: i for i, c in enumerate(cats)}
    int2idx = {it: i for i, it in enumerate(intents)}

    # Vocab
    vocab = Vocab(max_tokens=8000)
    vocab.fit(train_df["clean_question"])

    # Datasets
    train_ds = MultiTaskDataset(train_df, vocab, cat2idx, int2idx)
    val_ds = MultiTaskDataset(val_df, vocab, cat2idx, int2idx)
    test_ds = MultiTaskDataset(test_df, vocab, cat2idx, int2idx)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device("cpu")
    model = MultiTaskBiLSTM(
        vocab_size=len(vocab.w2i),
        embed_dim=128,
        hidden_dim=128,
        num_cats=len(cats),
        num_intents=len(intents),
        dropout=0.3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_score = 0.0
    history = []

    print("\nStarting BiLSTM Training (8 epochs)...")
    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        for x, dom, cat, intent in train_loader:
            x, dom, cat, intent = x.to(device), dom.to(device), cat.to(device), intent.to(device)
            optimizer.zero_grad()
            d_logits, c_logits, i_logits = model(x)
            
            l_dom = criterion(d_logits, dom)
            # Mask medical for cat and intent
            med_indices = (dom == 1).nonzero(as_tuple=True)[0]
            if len(med_indices) > 0:
                l_cat = criterion(c_logits[med_indices], cat[med_indices])
                l_int = criterion(i_logits[med_indices], intent[med_indices])
            else:
                l_cat = 0.0
                l_int = 0.0

            loss = l_dom + 1.2 * l_cat + 1.0 * l_int
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_eval = evaluate_model(model, val_loader, device, cats, intents)
        combined_f1 = (val_eval["category"]["macro_f1"] + val_eval["intent"]["macro_f1"]) / 2.0
        
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_cat_f1": val_eval["category"]["macro_f1"],
            "val_intent_f1": val_eval["intent"]["macro_f1"]
        })
        print(f"Epoch {epoch}/8 - Loss: {avg_loss:.4f} | Val Cat F1: {val_eval['category']['macro_f1']:.4f} | Val Intent F1: {val_eval['intent']['macro_f1']:.4f}")

        if combined_f1 > best_val_score:
            best_val_score = combined_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm_multitask.pt"))

    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "bilstm_multitask.pt")))
    test_results = evaluate_model(model, test_loader, device, cats, intents)
    print("\nBiLSTM Final Test Set Evaluation:")
    print("Domain:", test_results["domain"])
    print("Category:", test_results["category"])
    print("Intent:", test_results["intent"])

    # Save artifacts
    with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
        json.dump(vocab.w2i, f)
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump({"categories": cats, "intents": intents}, f, indent=2)
    with open(os.path.join(MODEL_DIR, "bilstm_metrics.json"), "w") as f:
        json.dump({
            "test_metrics": test_results,
            "history": history,
            "architecture": "Word Embedding (128) -> 2-layer BiLSTM (128) -> Max/Mean Pool -> LayerNorm Dense (256) -> Multi-Task Heads"
        }, f, indent=2)

    print(f"BiLSTM artifacts saved successfully to {MODEL_DIR}")
    return test_results

if __name__ == "__main__":
    train_bilstm()
