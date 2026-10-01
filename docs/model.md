# MedClass AI — Model Architecture & Empirical Benchmarks

## 1. Model Progression Strategy

To establish clear empirical rigor for the hackathon, MedClass AI implements and benchmarks three generations of NLP architectures evaluated on the identical held-out test split of 513 records:

| Generation | Architecture | Features / Backbone | Parameters |
|---|---|---|---|
| **Gen 1: Traditional ML** | Calibrated Linear SVM | TF-IDF (1-2 n-grams, 10,000 features) | ~35,000 |
| **Gen 2: Deep Learning** | BiLSTM Multi-Task | Word Embeddings (128) + 2-layer BiLSTM (128) | ~1.35 M |
| **Gen 3: Transformer SOTA** | Contextual Transformer | BERT-Tiny Encoder + Multi-Task Projections | ~4.40 M |

---

## 2. Empirical Benchmark Comparison (Test Set)

| Task | Metric | Baseline (Linear SVM) | Deep Learning (BiLSTM) | Transformer Multi-Task |
|---|---|---|---|---|
| **Domain (Medical vs Non-Med)** | Accuracy | 100.0% | 99.61% | **100.0%** |
| | Macro F1 | 1.0000 | 0.9864 | **1.0000** |
| **Category Classification** | Accuracy | 95.34% | 92.16% | **96.40%** |
| | Macro F1 | 0.9528 | 0.9310 | **0.9601** |
| | Weighted F1 | 0.9532 | 0.9245 | **0.9635** |
| **Intent Classification** | Accuracy | 98.09% | 98.73% | **99.15%** |
| | Macro F1 | 0.9495 | 0.9254 | **0.9769** |
| | Weighted F1 | 0.9803 | 0.9854 | **0.9914** |
| **Inference Latency (CPU)** | Latency | ~1.2 ms | ~4.8 ms | ~11.5 ms |

---

## 3. Multi-Task Transformer Design

The Contextual Transformer architecture fine-tunes a shared BERT backbone to simultaneously solve three clinical objectives:

1. **Input Encoding:** BERT subword tokenization with sequence length $L = 48$.
2. **Contextual Encoding:** 2-layer bidirectional self-attention blocks extracting contextual representation $H \in \mathbb{R}^{B \times L \times 128}$.
3. **Adaptive Pooling:** Mask-weighted mean pooling across non-padded tokens.
4. **Shared Dense Projection:** $\text{LayerNorm}(\text{GELU}(W_p \cdot \bar{h} + b_p)) \in \mathbb{R}^{B \times 256}$.
5. **Specialized Heads:**
   - Domain Head: $\mathbb{R}^{256} \to \mathbb{R}^2$
   - Category Head: $\mathbb{R}^{256} \to \mathbb{R}^{13}$
   - Intent Head: $\mathbb{R}^{256} \to \mathbb{R}^{10}$

### Multi-Task Objective:
$$\mathcal{L} = \mathcal{L}_{domain} + 1.2 \cdot \mathcal{L}_{category} + 1.0 \cdot \mathcal{L}_{intent}$$

Optimized using AdamW with weight decay $10^{-2}$, initial learning rate $5 \times 10^{-4}$, and linear warmup schedule.
