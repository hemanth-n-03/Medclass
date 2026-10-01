"""
FastAPI Route Handlers for MedClass AI
"""

import os
import json
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.question import (
    ClassifyRequest,
    ClassifyResponse,
    SimilarQuestionRequest,
    HealthResponse,
    AnalyticsSummaryResponse
)
from backend.app.services.classifier_service import classifier_service
from backend.app.services.analytics_service import analytics_service

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
COMPARISON_FILE = os.path.join(BASE_DIR, "ml", "evaluation", "model_comparison.json")
DATASET_STATS = os.path.join(BASE_DIR, "datasets", "dataset_stats.json")

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    classifier_service.initialize()
    pred = classifier_service.predictor
    return HealthResponse(
        status="healthy",
        service="MedClass AI Engine",
        version="2.0.0",
        model_loaded=pred.is_loaded if pred else False,
        device=str(pred.device) if pred else "cpu",
        supported_categories=len(pred.categories) if pred else 0,
        supported_intents=len(pred.intents) if pred else 0
    )

@router.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify_question(payload: ClassifyRequest):
    q = payload.question.strip()
    if not q or len(q) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The submitted inquiry is too short. Please provide at least 2 characters."
        )
    try:
        result = classifier_service.classify_question(q)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification inference error: {str(e)}"
        )

@router.post("/similar-questions", tags=["Search & Knowledge Retrieval"])
def get_similar_questions(payload: SimilarQuestionRequest):
    q = payload.question.strip()
    if not q or len(q) < 2:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query too short")
    try:
        results = classifier_service.find_similar(q, top_k=payload.top_k)
        return {"query": q, "count": len(results), "similar_questions": results}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/model-info", tags=["Model Metadata"])
def get_model_info():
    comparison_data = {}
    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE, "r") as f:
            comparison_data = json.load(f)

    stats_data = {}
    if os.path.exists(DATASET_STATS):
        with open(DATASET_STATS, "r") as f:
            stats_data = json.load(f)

    classifier_service.initialize()
    pred = classifier_service.predictor

    return {
        "model_name": "MedClass AI Multi-Task Contextual Transformer",
        "backbone": "BERT Contextual Encoder",
        "categories": pred.categories if pred else [],
        "intents": pred.intents if pred else [],
        "benchmarks": comparison_data,
        "dataset_summary": {
            "source": "NIH MedQuAD + OpenDomain Inquiries",
            "total_records": stats_data.get("total_records", 9256),
            "train_samples": stats_data.get("train_count", 7404),
            "val_samples": stats_data.get("val_count", 926),
            "test_samples": stats_data.get("test_count", 926),
            "license": "NIH Open Access / Research Use"
        }
    }

@router.get("/analytics", tags=["Auditing & Analytics"])
def get_analytics():
    return analytics_service.get_summary()

@router.get("/categories", tags=["Taxonomy"])
def list_categories():
    classifier_service.initialize()
    return {"categories": classifier_service.predictor.categories if classifier_service.predictor else []}

@router.get("/intents", tags=["Taxonomy"])
def list_intents():
    classifier_service.initialize()
    return {"intents": classifier_service.predictor.intents if classifier_service.predictor else []}
