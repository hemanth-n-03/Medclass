# MedClass AI — REST API Documentation

MedClass AI provides a RESTful API powered by FastAPI with Pydantic validation, CORS middleware, and error handling.

## Base URL
`http://127.0.0.1:8000/api`

---

## 1. Classify Inquiry
**Endpoint:** `POST /api/classify`  
**Description:** Accepts natural-language inquiries and performs domain verification, specialty category classification, clinical intention extraction, uncertainty assessment, emergency red flag detection, and similar question retrieval.

### Request Body:
```json
{
  "question": "What are the early symptoms of diabetes?"
}
```

### Response (200 OK):
```json
{
  "question": "What are the early symptoms of diabetes?",
  "is_medical": true,
  "domain_confidence": 0.9998,
  "category": "Endocrinology",
  "category_confidence": 0.9932,
  "secondary_categories": [
    { "category": "General Medicine", "confidence": 0.005 }
  ],
  "intent": "Symptoms",
  "intent_confidence": 0.9971,
  "secondary_intents": [],
  "overall_confidence": 0.9951,
  "uncertain": false,
  "uncertainty_reason": null,
  "explanation": {
    "explanation_text": "Question analyzed by MedClass Multi-Task Transformer...",
    "salient_tokens": ["early", "symptoms", "diabetes"],
    "methodology": "Multi-Task Contextual Attention & Feature Attribution"
  },
  "similar_questions": [
    {
      "question": "What are the symptoms of Diabetes mellitus type 1?",
      "similarity_score": 0.408,
      "answer_snippet": "The symptoms of diabetes include...",
      "source": "NIDDK",
      "focus_area": "Diabetes"
    }
  ],
  "routing": {
    "recommended_specialist": "Endocrinologist",
    "clinical_department": "Endocrinology, Diabetes & Metabolism",
    "suggested_action": "Inquire with a certified Endocrinologist..."
  },
  "emergency": {
    "is_emergency": false,
    "warning": null,
    "matched_indicators": []
  },
  "safety_disclaimer": "This system classifies medical questions and is not a medical diagnostic tool..."
}
```

---

## 2. Health & Engine Status
**Endpoint:** `GET /api/health`

### Response (200 OK):
```json
{
  "status": "healthy",
  "service": "MedClass AI Engine",
  "version": "2.0.0",
  "model_loaded": true,
  "device": "cpu",
  "supported_categories": 13,
  "supported_intents": 10
}
```

---

## 3. Query Analytics & Audit
**Endpoint:** `GET /api/analytics`

### Response (200 OK):
Returns KPI totals, category distributions, intent distributions, and live audit feed events.

---

## 4. Model Architecture & Benchmarks
**Endpoint:** `GET /api/model-info`

Returns architecture definitions, dataset splits, and empirical evaluation metrics across models.

---

## 5. Similar Question Search
**Endpoint:** `POST /api/similar-questions`

### Request:
```json
{
  "question": "How to prevent heart disease?",
  "top_k": 3
}
```
