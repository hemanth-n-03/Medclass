# MedClass AI — Hackathon Pitch Deck & Speaker Notes

## Slide 1: Title Slide
**Slide Title:** MedClass AI: Intelligent Medical Question Understanding & Classification  
**Subtitle:** Autonomous Deep Learning & Multi-Task Transformers for Clinical Question Intelligence  
**Presenter Notes:**  
"Good morning, judges. Today we present MedClass AI—an autonomous medical question intelligence platform designed to understand, categorize, and triage healthcare inquiries using state-of-the-art multi-task deep learning."

---

## Slide 2: The Problem
**Slide Title:** Natural-Language Inquiries Overwhelm Healthcare Portals  
**Key Points:**
- Patients ask thousands of unstructured, conversational health questions daily.
- Existing hospital routing systems rely on rigid keyword searches or blunt triage.
- Misrouted inquiries delay patient care, cause clinical burnout, and lead to poor information retrieval.  
**Presenter Notes:**  
"When patients interact with digital health systems, their questions are nuanced: 'What are early symptoms of diabetes?' isn't just about 'diabetes'—it's an inquiry seeking specific *symptoms* directed at *endocrinology*. Traditional search engines fail to understand clinical intent."

---

## Slide 3: Why It Matters
**Slide Title:** Classification Precedes Diagnosis & Safe Routing  
**Key Points:**
- **Inquiry Classification ≠ Medical Diagnosis:** Before answering a patient, an AI system must first safely understand *what kind* of question is being asked.
- Out-of-domain noise (tech questions, casual chatter) wastes clinical resources.
- Acute emergency questions (heart attacks, strokes) must be caught instantly and diverted to emergency hotlines.  
**Presenter Notes:**  
"Medical AI systems have often made the mistake of rushing to diagnose patients. We take a principled Responsible AI approach: Classify the *question*, understand its intent, estimate uncertainty, and route it to the right department—without ever making unsafe diagnostic claims."

---

## Slide 4: Our Solution
**Slide Title:** MedClass AI — End-to-End Clinical NLP Intelligence  
**Key Points:**
- **Stage 1:** Medical vs Non-Medical Domain Gate.
- **Stage 2:** Multi-Task Contextual Transformer classifying 13 Medical Specialties & 10 Clinical Intentions.
- **Stage 3:** Calibrated Uncertainty Scoring & Feature Attribution.
- **Stage 4:** Semantic Cosine Similarity Search over 4,000+ verified NIH MedQuAD questions.  
**Presenter Notes:**  
"MedClass AI is a full-stack, production-grade intelligence pipeline that performs domain detection, simultaneous category and intention prediction, uncertainty flagging, emergency triage, and institutional knowledge retrieval."

---

## Slide 5: System Architecture
**Slide Title:** Modular Multi-Task Architecture  
**Key Points:**
- Next.js 14 interactive UI with animated step-by-step reasoning simulation.
- FastAPI asynchronous backend serving sub-20ms inference.
- Shared BERT contextual encoder with joint task-specific projection heads.  
**Presenter Notes:**  
"Our architecture separates concerns cleanly. The user interacts with a modern Next.js interface, while our FastAPI backend handles requests with an in-memory pre-warmed PyTorch model. A single forward pass yields domain verification, category classification, and intention extraction."

---

## Slide 6: Dataset Engineering
**Slide Title:** Curated NIH MedQuAD Corpus  
**Key Points:**
- 5,125 total samples sourced from NIH (NINDS, NHLBI, NIDDK, CancerGov, CDC) plus contrastive open-domain questions.
- 1,427 duplicate templates purged; balanced across 13 specialties.
- Zero-leakage 80/10/10 stratified split strictly verified.  
**Presenter Notes:**  
"Rather than using synthetic or hallucinated data, we built our dataset from the NIH MedQuAD corpus across 8 major national health institutes, augmented with contrastive open-domain questions, guaranteeing zero train-test data leakage."

---

## Slide 7: Deep Learning Evolution
**Slide Title:** Progression Across 3 Model Generations  
**Key Points:**
- **Gen 1:** TF-IDF + Calibrated Linear SVM (Baseline ML).
- **Gen 2:** 2-Layer Bidirectional LSTM with adaptive pooling (Deep Learning).
- **Gen 3:** Multi-Task BERT Contextual Transformer (State-of-the-Art).  
**Presenter Notes:**  
"In the spirit of rigorous engineering, we implemented and trained three complete generations: from traditional TF-IDF SVM, to recurrent BiLSTMs, up to modern multi-task transformers, recording genuine empirical metrics on the identical test split."

---

## Slide 8: Multi-Task Classification
**Slide Title:** Shared Contextual Representations  
**Key Points:**
- Joint loss optimization: $\mathcal{L}_{total} = \mathcal{L}_{domain} + 1.2 \cdot \mathcal{L}_{cat} + 1.0 \cdot \mathcal{L}_{int}$.
- Prevents catastrophic forgetting and regularizes representations across tasks.
- Produces secondary candidate categories for multi-label clinical contexts.  
**Presenter Notes:**  
"By using multi-task learning, the model shares contextual representations between category and intent. Learning what constitutes a symptom helps the model understand endocrinology, leading to higher generalization."

---

## Slide 9: Benchmark Results
**Slide Title:** Transformer Outperforms Across All Tasks  
**Key Points:**
- **Domain Gate:** 100.0% Accuracy, 1.0000 Macro F1.
- **Category:** 96.40% Accuracy, 0.9601 Macro F1 (vs 92.16% BiLSTM, 95.34% SVM).
- **Intent:** 99.15% Accuracy, 0.9769 Macro F1 (vs 98.73% BiLSTM, 98.09% SVM).
- **Latency:** ~11.5 ms on CPU.  
**Presenter Notes:**  
"The Transformer model achieves 96.4% category accuracy and 99.15% intent accuracy on unseen test data, with an inference speed of approximately 11 milliseconds on CPU."

---

## Slide 10: Live Product Demo
**Slide Title:** Live Demonstration Walkthrough  
**Scenarios to Demonstrate:**
1. *Standard Clinical Query:* 'What are the early symptoms of diabetes?' → Endocrinology / Symptoms (99% conf) + similar NIH cases.
2. *Out-of-Domain Query:* 'How do I repair my laptop battery?' → Domain filter active: Outside medical domain.
3. *Emergency Red Flag:* 'I have crushing chest pain radiating to my arm' → Critical Emergency Alert triggered.
4. *Ambiguous Query:* 'diabetes?' → Flagged as Low Confidence / Uncertain.  
**Presenter Notes:**  
"Let us demonstrate the live application. Notice how effortlessly it handles clinical inquiries, gracefully rejects out-of-domain tech chatter, and instantly flags emergency red flags."

---

## Slide 11: Responsible AI & Patient Safety
**Slide Title:** Safety by Design  
**Key Points:**
- Explicit non-diagnostic disclaimer on all views.
- Immediate emergency detection routing to 911 / 112.
- Uncertainty disclosure prevents unwarranted AI confidence.
- Zero storage of Protected Health Information (PHI).  
**Presenter Notes:**  
"Safety is our first priority. MedClass AI never pretends to be a doctor. It states its uncertainty openly and directs acute symptoms to real-world emergency care."

---

## Slide 12: Future Scope & Conclusion
**Slide Title:** Roadmap & Conclusion  
**Key Points:**
- Multilingual clinical tokenizer support (Spanish, Hindi, French).
- Integration into electronic health record (EHR) patient portal triage inboxes.
- Active human-in-the-loop clinical feedback refinement.  
**Presenter Notes:**  
"MedClass AI represents a dependable, verified foundation for medical question understanding. Thank you, and we welcome your questions!"
