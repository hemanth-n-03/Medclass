"""
Machine Learning Pipeline Unit Tests
Validates data preprocessing, ontology mapping, similarity search, confidence/uncertainty logic, and emergency detection
"""

import pytest
from ml.preprocessing.build_dataset import clean_question_text, extract_intent, map_category
from ml.inference.similarity_engine import SimilarityEngine
from ml.inference.predictor import get_predictor

def test_clean_question_text():
    raw = "What is (are) Glaucoma ? ?"
    cleaned = clean_question_text(raw)
    assert cleaned == "What is Glaucoma?"
    assert clean_question_text("  How to prevent   Asthma ? ") == "How can you prevent Asthma?"

def test_extract_intent():
    assert extract_intent("What are the symptoms of diabetes?") == "Symptoms"
    assert extract_intent("What causes coronary artery disease?") == "Causes"
    assert extract_intent("What are the treatments for hypertension?") == "Treatment"
    assert extract_intent("How is leukemia diagnosed?") == "Diagnosis"
    assert extract_intent("How can you prevent heart disease?") == "Prevention"
    assert extract_intent("Who is at risk for osteoporosis?") == "Risk Factors"
    assert extract_intent("What are the side effects of penicillin?") == "Side Effects"
    assert extract_intent("What is the proper dosage for ibuprofen?") == "Medication"

def test_map_category():
    assert map_category({"focus_area": "Heart Attack", "source": "NHLBI", "question": ""}) == "Cardiology"
    assert map_category({"focus_area": "Alzheimer's Disease", "source": "NINDS", "question": ""}) == "Neurology"
    assert map_category({"focus_area": "Asthma", "source": "NHLBI", "question": ""}) == "Respiratory"
    assert map_category({"focus_area": "Type 2 Diabetes", "source": "NIDDK", "question": ""}) == "Endocrinology"
    assert map_category({"focus_area": "Breast Cancer", "source": "CancerGov", "question": ""}) == "Oncology"
    assert map_category({"focus_area": "Eczema", "source": "NIAMS", "question": ""}) == "Dermatology"

def test_similarity_engine():
    engine = SimilarityEngine()
    engine.load()
    results = engine.search("What are diabetes symptoms?", top_k=3)
    assert len(results) > 0
    assert "question" in results[0]
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] >= 0.0

def test_emergency_detector():
    pred = get_predictor()
    res_emer = pred.check_emergency("I have severe crushing chest pain radiating to my left arm")
    assert res_emer["is_emergency"] is True
    assert len(res_emer["matched_indicators"]) > 0

    res_safe = pred.check_emergency("What causes mild skin dryness during winter?")
    assert res_safe["is_emergency"] is False

def test_predictor_uncertainty():
    pred = get_predictor()
    res = pred.predict("What are the early symptoms of diabetes?")
    assert res["is_medical"] is True
    assert res["uncertain"] is False
    assert res["overall_confidence"] > 0.80
