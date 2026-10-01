"""
Pydantic Schemas for MedClass AI Requests & Responses
Supports multi-symptom clinical entities, combination emergency triage,
proportional next steps, and Indian Standard Time (IST) metadata.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ClassifyRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Natural language medical or clinical inquiry")

class SecondaryCandidate(BaseModel):
    category: Optional[str] = None
    intent: Optional[str] = None
    confidence: float
    reason: Optional[str] = None

class EmergencyInfo(BaseModel):
    is_emergency: bool
    safety_level: Optional[str] = "routine"
    warning: Optional[str] = None
    matched_indicators: List[str] = []

class ExplanationInfo(BaseModel):
    explanation_text: str
    salient_tokens: List[str]
    methodology: str

class SimilarQuestion(BaseModel):
    question: str
    similarity_score: float
    answer_snippet: Optional[str] = None
    source: Optional[str] = None
    source_organization: Optional[str] = None
    focus_area: Optional[str] = None

class ClinicalSignal(BaseModel):
    entity: str
    domain: str
    signal: str
    confidence: Optional[float] = None

class RoutingInfo(BaseModel):
    recommended_specialist: str
    clinical_department: str
    suggested_action: str

class ClassifyResponse(BaseModel):
    question: str
    normalized_question: Optional[str] = None
    corrections: List[Dict[str, str]] = []
    is_medical: bool
    domain_confidence: float
    message: Optional[str] = None
    
    # Backward-compatible category fields
    category: Optional[str] = None
    category_confidence: float = 0.0
    primary_category: Optional[str] = None
    primary_category_confidence: Optional[float] = None
    primary_category_reason: Optional[str] = None
    secondary_categories: List[Dict[str, Any]] = []

    # Intent fields
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    secondary_intents: List[Dict[str, Any]] = []

    # Confidence and Uncertainty
    overall_confidence: float = 0.0
    uncertain: bool = False
    uncertainty_reason: Optional[str] = None
    clarification_needed: Optional[str] = None

    # Multi-Symptom & Entity Recognition
    is_multi_symptom: bool = False
    detected_entities: List[str] = []
    clinical_signals: List[Dict[str, Any]] = []

    # Guidance and Routing
    safety_level: Optional[str] = "routine"
    suggested_next_steps: List[str] = []
    explanation: Optional[ExplanationInfo] = None
    similar_questions: List[SimilarQuestion] = []
    routing: Optional[RoutingInfo] = None
    emergency: Optional[EmergencyInfo] = None
    safety_disclaimer: str

    # Localization
    timestamp_ist: Optional[str] = None

class SimilarQuestionRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    device: str
    supported_categories: int
    supported_intents: int

class AnalyticsSummaryResponse(BaseModel):
    total_queries: int
    medical_queries: int
    non_medical_queries: int
    uncertain_queries: int
    emergency_queries: int
    avg_confidence: float
    category_distribution: Dict[str, int]
    intent_distribution: Dict[str, int]
    recent_activity: List[Dict[str, Any]]
