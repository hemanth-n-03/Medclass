"""
Classifier Service wrapping predictor and analytics logging
"""

from ml.inference.predictor import get_predictor
from backend.app.services.analytics_service import analytics_service

class ClassifierService:
    def __init__(self):
        self.predictor = None

    def initialize(self):
        if self.predictor is None:
            self.predictor = get_predictor()

    def classify_question(self, question: str) -> dict:
        self.initialize()
        result = self.predictor.predict(question)
        # Log to analytics
        analytics_service.record_query(result)
        return result

    def find_similar(self, question: str, top_k: int = 4) -> list:
        self.initialize()
        return self.predictor.sim_engine.search(question, top_k=top_k)

classifier_service = ClassifierService()
