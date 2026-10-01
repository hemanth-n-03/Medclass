"""
Backend API Unit and Integration Tests
Tests all endpoints, validation logic, edge cases, multi-symptom queries,
combination emergency detection with Indian helpline (112 / 108), and IST timestamp auditing.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["supported_categories"] > 0
    assert data["supported_intents"] > 0

def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "categories" in data
    assert "intents" in data
    assert "benchmarks" in data
    assert len(data["categories"]) >= 10

def test_analytics_endpoint():
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "category_distribution" in data
    assert "intent_distribution" in data
    assert "recent_activity" in data

def test_classify_medical_question():
    payload = {"question": "What are the early symptoms of diabetes?"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_medical"] is True
    assert data["category"] == "Endocrinology"
    assert data["intent"] == "Symptoms"
    assert data["category_confidence"] > 0.80
    assert data["intent_confidence"] > 0.80
    assert "safety_disclaimer" in data
    assert len(data["similar_questions"]) > 0
    assert "timestamp_ist" in data
    assert "IST" in data["timestamp_ist"]

def test_classify_neurology_question():
    payload = {"question": "What causes migraine headaches?"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_medical"] is True
    assert data["category"] == "Neurology"
    assert data["intent"] == "Causes"

def test_classify_cardiology_question():
    payload = {"question": "How to prevent heart disease?"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_medical"] is True
    assert data["category"] == "Cardiology"

def test_classify_multi_symptom_question():
    payload = {"question": "I am having leg pain, fever and chest pain"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_medical"] is True
    assert data["is_multi_symptom"] is True
    assert "Chest pain" in data["detected_entities"]
    assert "Fever" in data["detected_entities"]
    assert "Leg pain" in data["detected_entities"]
    assert len(data["clinical_signals"]) >= 3
    assert data["primary_category"] == "Cardiology"
    assert "Cardiology was prioritized" in data["primary_category_reason"]
    assert len(data["secondary_categories"]) >= 2
    assert len(data["suggested_next_steps"]) > 0

def test_combination_emergency_detection():
    payload = {"question": "I have severe chest pain and difficulty breathing"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["emergency"]["is_emergency"] is True
    assert data["safety_level"] == "critical"
    warning = data["emergency"]["warning"]
    assert "112" in warning
    assert "108" in warning
    assert "911" not in warning

def test_ambiguous_short_question():
    payload = {"question": "Headache"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_medical"] is True
    assert data["uncertain"] is True
    assert data["clarification_needed"] is not None
    assert "Headache" in data["clarification_needed"]

def test_classify_non_medical_question():
    payload = {"question": "How do I repair my laptop battery?"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_medical"] is False
    assert "outside the supported medical domain" in data["message"]

def test_empty_question_validation():
    payload = {"question": ""}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 422

def test_similar_questions_endpoint():
    payload = {"question": "What causes diabetes?", "top_k": 3}
    response = client.post("/api/similar-questions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "similar_questions" in data
    assert len(data["similar_questions"]) <= 3

def test_taxonomy_endpoints():
    res_cats = client.get("/api/categories")
    assert res_cats.status_code == 200
    assert len(res_cats.json()["categories"]) >= 10

    res_intents = client.get("/api/intents")
    assert res_intents.status_code == 200
    assert len(res_intents.json()["intents"]) >= 8

def test_conversational_equivalent_thirst_and_urination_queries():
    equivalent_queries = [
        "I have been thirsty all day and going to the bathroom constantly, what could be wrong?",
        "I feel extremely thirsty and keep urinating even though I haven't been drinking much water",
        "I've suddenly started drinking lots of water and waking up several times at night to pee"
    ]
    for q in equivalent_queries:
        res = client.post("/api/classify", json={"question": q})
        assert res.status_code == 200
        data = res.json()
        assert data["is_medical"] is True, f"Failed domain gate for: {q}"
        assert data["category"] == "Endocrinology", f"Failed category for: {q}"
        assert data["is_multi_symptom"] is True
        assert "Excessive thirst (Polydipsia)" in data["detected_entities"]
        assert "Frequent urination (Polyuria)" in data["detected_entities"]
        assert "glucose" in data["explanation"]["explanation_text"].lower() or "blood sugar" in data["primary_category_reason"].lower()
        # Verify non-diagnostic framing
        explanation_text = data["explanation"]["explanation_text"].lower()
        assert "cannot diagnose" in explanation_text or "not a medical diagnostic" in data["safety_disclaimer"].lower()

def test_misspelled_medical_queries():
    test_cases = [
        ("wat r the signs of diabetis", "Endocrinology", "Symptoms"),
        ("gloucoma", "General Medicine", None),
        ("migrene", "Neurology", None),
        ("astma", "Respiratory", None),
    ]
    for raw_q, expected_cat, expected_intent in test_cases:
        res = client.post("/api/classify", json={"question": raw_q})
        assert res.status_code == 200
        data = res.json()
        assert data["is_medical"] is True
        if expected_cat:
            assert data["category"] == expected_cat
        if expected_intent:
            assert data["intent"] == expected_intent
        assert "normalized_question" in data
        assert data["normalized_question"] is not None

def test_informal_symptom_descriptions():
    cases = [
        ("head hurts one side and light makes it worse", "Neurology"),
        ("heart feels like it is skipping beats when I lie down", "Cardiology"),
        ("my skin has red patches that itch like crazy", "Dermatology"),
        ("can't catch my breath after walking up stairs", "Respiratory"),
    ]
    for q, expected_cat in cases:
        res = client.post("/api/classify", json={"question": q})
        assert res.status_code == 200
        data = res.json()
        assert data["is_medical"] is True
        assert data["category"] == expected_cat

def test_non_medical_negative_queries():
    negative_queries = [
        "how do I reset my wifi router",
        "best places to visit in Japan",
        "how to bake sourdough bread",
        "what is python asyncio",
        "why is my car making a rattling noise"
    ]
    for q in negative_queries:
        res = client.post("/api/classify", json={"question": q})
        assert res.status_code == 200
        data = res.json()
        assert data["is_medical"] is False
        assert "outside the supported medical domain" in data["message"]

