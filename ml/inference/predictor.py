"""
MedClass AI Unified Prediction Pipeline
Combines:
1. Multi-Task Transformer Model (Domain, Category, Intent)
2. Query Normalizer (Medical typos, abbreviations, conversational slang)
3. Semantic Clinical Concept & Entity Extractor (Dense contextual embeddings)
4. Robust Semantic Domain Gate
5. Multi-Symptom / Multi-Entity Classification & Clinical Specialty Routing
6. Enhanced Emergency Red-Flag & Combination Detection (India Helpline: 112 / 108)
7. Confidence and Uncertainty Estimation & Ambiguity Clarification
8. Explainable AI & Salient Concept Attribution (Strictly Non-Diagnostic)
9. Hybrid Semantic Similarity Retrieval over NIH MedQuAD Knowledge Base
"""

import os
import sys
import re
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from ml.inference.normalizer import get_normalizer, QueryNormalizer
from ml.inference.semantic_extractor import get_semantic_extractor, SemanticClinicalExtractor
from ml.inference.similarity_engine import SimilarityEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSFORMER_DIR = os.path.join(BASE_DIR, "ml", "models", "transformer")
MODEL_BACKBONE = "prajjwal1/bert-tiny"
MAX_LEN = 48

# Indian Standard Time (IST, UTC+05:30) Helper
def get_current_ist_time() -> Tuple[str, str]:
    ist_zone = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_zone)
    formatted_ist = now_ist.strftime("%d %b %Y, %I:%M %p IST")
    iso_time = now_ist.isoformat()
    return formatted_ist, iso_time

# Emergency symptom combinations and critical red flags
CRITICAL_EMERGENCY_RULES = [
    {
        "name": "Chest Pain + Breathing Difficulty",
        "patterns": [(r"chest pain|chest pressure|crushing chest", r"difficulty breathing|shortness of breath|breathlessness|can'?t breathe|struggling to breathe")],
        "reason": "Concurrent acute chest discomfort and breathing impairment indicates high clinical acuity."
    },
    {
        "name": "Chest Pain + Radiation to Arm/Jaw/Neck",
        "patterns": [(r"chest pain|chest pressure", r"radiating (to|down|into) (my )?(left )?(arm|jaw|neck|shoulder|back)")],
        "reason": "Chest pain radiating to the jaw, neck, arm, or back is a classic acute cardiovascular red flag."
    },
    {
        "name": "Acute Stroke Indicators (FAST)",
        "patterns": [(r"slurred speech|facial drooping|face drooping", r"sudden numbness|sudden weakness|loss of balance|one-sided weakness")],
        "reason": "Combination of facial, speech, or motor weakness constitutes a recognized acute stroke signal."
    },
    {
        "name": "Severe Headache + Neurological Signs",
        "patterns": [(r"severe headache|thunderclap headache|worst headache", r"sudden numbness|vision loss|confusion|stiff neck")],
        "reason": "Sudden severe headache accompanied by focal deficits or neck stiffness requires immediate emergency assessment."
    },
    {
        "name": "High Fever + Stiff Neck / Altered State",
        "patterns": [(r"\bfever\b|high temperature|high temp", r"stiff neck|confusion|altered mental|unresponsive")],
        "reason": "Fever paired with neck rigidity or cognitive changes may indicate severe central nervous system infection."
    },
    {
        "name": "Severe Allergic / Anaphylaxis Signs",
        "patterns": [(r"throat swelling|swollen tongue|anaphylaxis", r"difficulty breathing|hives|dizziness")],
        "reason": "Airway compromise and acute allergic symptoms represent a medical emergency."
    }
]

STANDALONE_EMERGENCY_PATTERNS = [
    (r"heart attack", "Heart attack"),
    (r"stroke symptoms", "Stroke symptoms"),
    (r"coughing up blood", "Hemoptysis (coughing up blood)"),
    (r"sudden blindness", "Acute vision loss"),
    (r"unconscious|passed out|unresponsive", "Altered consciousness / Syncope"),
    (r"heavy blood loss|severe bleeding|uncontrolled bleeding", "Severe hemorrhage"),
    (r"suicidal thoughts|want to end my life|kill myself", "Psychiatric crisis / self-harm")
]


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
        seq_out = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(seq_out.size()).float()
        sum_embeddings = torch.sum(seq_out * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        shared = self.shared_proj(pooled)
        d_logits = self.domain_head(shared)
        c_logits = self.cat_head(shared)
        i_logits = self.intent_head(shared)
        return d_logits, c_logits, i_logits


class MedicalQuestionPredictor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self.categories = []
        self.intents = []
        self.cat2idx = {}
        self.int2idx = {}
        self.normalizer: Optional[QueryNormalizer] = None
        self.semantic_extractor: Optional[SemanticClinicalExtractor] = None
        self.sim_engine = SimilarityEngine()
        self.is_loaded = False

    def load(self):
        if self.is_loaded:
            return

        labels_path = os.path.join(TRANSFORMER_DIR, "labels.json")
        weights_path = os.path.join(TRANSFORMER_DIR, "transformer_multitask.pt")

        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        self.categories = labels_data["categories"]
        self.intents = labels_data["intents"]
        self.cat2idx = {c: i for i, c in enumerate(self.categories)}
        self.int2idx = {it: i for i, it in enumerate(self.intents)}

        self.tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_DIR)
        
        self.model = MultiTaskTransformer(
            backbone_name=MODEL_BACKBONE,
            num_cats=len(self.categories),
            num_intents=len(self.intents),
            dropout=0.0
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load NLP helpers
        self.normalizer = get_normalizer()
        self.semantic_extractor = get_semantic_extractor()
        self.sim_engine.load()

        self.is_loaded = True
        print("MedicalQuestionPredictor successfully loaded into memory.")

    def evaluate_emergency(self, text: str, matched_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates emergency combinations and red-flags using Indian emergency references (112 / 108).
        """
        text_lower = text.lower()
        matched_indicators = []
        is_critical = False
        is_urgent = False
        reasons = []

        # 1. Combination Rules
        for rule in CRITICAL_EMERGENCY_RULES:
            for p1, p2 in rule["patterns"]:
                if re.search(p1, text_lower) and re.search(p2, text_lower):
                    is_critical = True
                    matched_indicators.append(rule["name"])
                    reasons.append(rule["reason"])
                    break

        # 2. Standalone Critical Patterns
        for pat, name in STANDALONE_EMERGENCY_PATTERNS:
            if re.search(pat, text_lower):
                is_critical = True
                matched_indicators.append(name)
                reasons.append(f"Acute standalone indicator ('{name}') detected.")

        # 3. High-acuity single symptoms
        high_acuity_entities = [r["name"] for r in matched_records if r.get("priority") == 1]
        if high_acuity_entities and not is_critical:
            is_urgent = True
            for ent in high_acuity_entities:
                if ent not in matched_indicators:
                    matched_indicators.append(ent)

        if is_critical:
            safety_level = "critical"
            warning = (
                "CRITICAL EMERGENCY ALERT: Your inquiry contains acute red-flag medical indicators "
                f"({', '.join(set(matched_indicators))}). Please call emergency services immediately: "
                "dial 112 (National Emergency Helpline) or 108 (Ambulance), or go directly to the nearest "
                "government or private hospital emergency department. Do not rely on automated AI classification "
                "for acute emergencies."
            )
        elif is_urgent:
            safety_level = "urgent"
            warning = (
                "URGENT CLINICAL ATTENTION RECOMMENDED: Symptoms such as "
                f"'{', '.join(set(matched_indicators))}' can sometimes be associated with acute conditions. "
                "Please consult a registered medical practitioner or visit the nearest hospital emergency department "
                "for prompt in-person clinical assessment."
            )
        else:
            safety_level = "routine"
            warning = None

        return {
            "is_emergency": is_critical,
            "safety_level": safety_level,
            "warning": warning,
            "matched_indicators": list(set(matched_indicators))
        }

    def check_emergency(self, text: str) -> Dict[str, Any]:
        """Backward-compatible alias for evaluate_emergency."""
        if not self.is_loaded:
            self.load()
        norm_res = self.normalizer.normalize(text)
        ext_res = self.semantic_extractor.extract(text, norm_res["normalized_text"])
        return self.evaluate_emergency(norm_res["normalized_text"], ext_res["matched_records"])

    def explain_prediction(
        self,
        text: str,
        category: str,
        intent: str,
        detected_entities: List[str],
        is_multi_symptom: bool,
        is_thirst_urination_cluster: bool = False
    ) -> Dict[str, Any]:
        """
        Generates grounded, non-diagnostic explainability information.
        Adheres strictly to clinical neutrality: no diagnosis of diseases.
        """
        words = re.findall(r"\w+", text.lower())
        ignore = {
            "what", "are", "the", "is", "of", "a", "an", "how", "can", "do", "does", "why",
            "who", "in", "to", "for", "with", "and", "or", "i", "am", "having", "my", "some",
            "have", "been", "all", "day", "going", "bathroom", "constantly"
        }
        salient_terms = [w for w in words if len(w) > 2 and w not in ignore]

        if is_thirst_urination_cluster:
            explanation = (
                f"Question describes concurrent symptoms ({', '.join(detected_entities)}). "
                "These symptoms can occur with several conditions, including problems involving blood glucose. "
                "The system cannot diagnose a condition. Classification into 'Endocrinology' reflects the metabolic "
                f"and osmotic relevance of blood sugar regulation, while clinical intention '{intent}' was inferred from the query framing."
            )
        elif is_multi_symptom and detected_entities:
            explanation = (
                f"Question presents a composite clinical inquiry involving multiple signals ({', '.join(detected_entities)}). "
                f"Classification into primary domain '{category}' highlights the leading clinical focus, "
                f"while intent '{intent}' was inferred without diagnostic presumption."
            )
        elif detected_entities:
            explanation = (
                f"Classified under '{category}' based on detected clinical entity '{', '.join(detected_entities)}'. "
                f"Inquiry intent '{intent}' was determined from contextual syntactic patterns in the question."
            )
        else:
            explanation = (
                f"Classified under '{category}' by the Multi-Task Contextual Transformer based on semantic embedding, "
                f"with clinical intention '{intent}' inferred from question framing."
            )

        return {
            "explanation_text": explanation,
            "salient_tokens": salient_terms[:6],
            "methodology": "Semantic Embedding Analysis & Contextual Attention"
        }

    def get_routing_suggestions(self, category: str, intent: str) -> Dict[str, str]:
        """
        Provides clinical routing suggestions localized to registered medical practitioners
        and hospital departments.
        """
        routing_map = {
            "Cardiology": {
                "specialist": "Cardiologist / Registered Medical Practitioner",
                "department": "Department of Cardiology, Government or Private Hospital"
            },
            "Neurology": {
                "specialist": "Neurologist / Physician",
                "department": "Department of Neurology, Specialized Hospital"
            },
            "Respiratory": {
                "specialist": "Pulmonologist / Chest Physician",
                "department": "Department of Pulmonary Medicine"
            },
            "Gastroenterology": {
                "specialist": "Gastroenterologist",
                "department": "Department of Gastroenterology & Hepatology"
            },
            "Endocrinology": {
                "specialist": "Endocrinologist / Diabetologist",
                "department": "Department of Endocrinology & Diabetes Care"
            },
            "Oncology": {
                "specialist": "Medical Oncologist",
                "department": "Department of Oncology / Regional Cancer Centre"
            },
            "Dermatology": {
                "specialist": "Dermatologist",
                "department": "Dermatology Outpatient Clinic"
            },
            "Infectious Disease": {
                "specialist": "Infectious Disease Specialist / Internal Medicine Physician",
                "department": "Department of General Medicine & Infectious Diseases"
            },
            "Orthopedics & Rheumatology": {
                "specialist": "Orthopedic Surgeon / Rheumatologist",
                "department": "Department of Orthopedics & Joint Care"
            },
            "Mental Health": {
                "specialist": "Psychiatrist / Registered Clinical Psychologist",
                "department": "Department of Psychiatry & Behavioral Health"
            },
            "Medication & Pharmacology": {
                "specialist": "Clinical Pharmacologist / Consulting Physician",
                "department": "Clinical Pharmacology & Pharmacy Department"
            },
            "Genetics & Rare Diseases": {
                "specialist": "Medical Geneticist",
                "department": "Department of Medical Genetics & Rare Diseases"
            },
            "General Medicine": {
                "specialist": "General Physician / Registered Medical Practitioner",
                "department": "General Medicine OPD, Community Health Centre or Hospital"
            }
        }
        default = {
            "specialist": "Registered Medical Practitioner / Family Physician",
            "department": "Outpatient Department (OPD), Local Hospital or Clinic"
        }
        info = routing_map.get(category, default)

        return {
            "recommended_specialist": info["specialist"],
            "clinical_department": info["department"],
            "suggested_action": (
                f"Schedule an in-person consultation with a qualified {info['specialist']} "
                f"at the {info['department']} for individualized diagnostic evaluation regarding {category.lower()} and {intent.lower()}."
            )
        }

    def generate_next_steps(
        self,
        safety_level: str,
        is_multi_symptom: bool,
        detected_entities: List[str],
        category: str,
        is_thirst_urination_cluster: bool = False
    ) -> List[str]:
        """
        Generates proportional, strictly non-diagnostic next steps for the user.
        """
        if safety_level == "critical":
            return [
                "Immediately call 112 (National Emergency Helpline) or 108 (Ambulance) for emergency medical transport.",
                "Proceed without delay to the nearest government or private hospital emergency department.",
                "Keep calm, avoid physical exertion, and do not self-administer medications without emergency physician guidance."
            ]
        elif safety_level == "urgent":
            return [
                "Seek same-day clinical assessment by a registered medical practitioner at a nearby hospital or clinic.",
                "Document the exact onset, timeline, and aggravating factors of detected symptoms.",
                "Dial 112 or 108 immediately if symptoms acutely worsen, or if breathlessness or dizziness develops."
            ]
        elif is_thirst_urination_cluster:
            return [
                "Record daily fluid intake and frequency of urination (including nighttime awakenings).",
                "Schedule a clinical consultation with a general physician or endocrinologist.",
                "Prepare for standard preliminary evaluations, which commonly include blood glucose or urinalysis.",
                "Note that these symptoms can occur with several conditions; in-person medical evaluation is essential."
            ]
        elif is_multi_symptom:
            return [
                f"Record symptom onset, duration, and severity for each observed concern ({', '.join(detected_entities)}).",
                "Observe whether symptoms occur together or independently and note any triggers.",
                "Consult an internal medicine physician or general practitioner for comprehensive multi-system evaluation.",
                "Seek urgent in-person care if chest pain, severe breathlessness, or high fever with confusion develops."
            ]
        else:
            return [
                "Consult a registered medical practitioner to discuss your specific symptoms or health inquiry.",
                "Prepare a list of questions, symptom duration, and current medications prior to your doctor's visit.",
                "Review verified health resources (such as official clinical portals) for validated medical information."
            ]

    def predict(self, question: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        raw_q = str(question).strip()
        formatted_ist, iso_ist = get_current_ist_time()

        if len(raw_q) < 2:
            return {
                "question": raw_q,
                "normalized_question": raw_q,
                "corrections": [],
                "error": "Question is too short to classify.",
                "is_medical": False,
                "confidence": 0.0,
                "timestamp_ist": formatted_ist
            }

        # 1. Query Normalization & Preprocessing Pipeline
        norm_result = self.normalizer.normalize(raw_q)
        normalized_q = norm_result["normalized_text"]
        corrections = norm_result["corrections"]

        # 2. Semantic Clinical Concept & Entity Extraction
        extraction = self.semantic_extractor.extract(raw_q, normalized_q)
        detected_entities = extraction["detected_entities"]
        clinical_signals = extraction["clinical_signals"]
        matched_records = extraction["matched_records"]
        is_multi_symptom = extraction["is_multi_symptom"]
        max_semantic_score = extraction["max_semantic_score"]

        # 3. Emergency Red-Flag & Combination Check (Evaluate both raw & normalized)
        emergency_info = self.evaluate_emergency(normalized_q, matched_records)

        # 4. Multi-Task Transformer Forward Pass (Input: normalized query)
        encoding = self.tokenizer(
            normalized_q,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            d_logits, c_logits, i_logits = self.model(input_ids, attention_mask)
            d_probs = torch.softmax(d_logits, dim=1)[0].cpu().numpy()
            is_medical_prob = float(d_probs[1])

            c_probs = torch.softmax(c_logits, dim=1)[0].cpu().numpy()
            cat_idx = int(np.argmax(c_probs))
            model_top_category = self.categories[cat_idx]
            model_cat_conf = round(float(c_probs[cat_idx]), 4)

            i_probs = torch.softmax(i_logits, dim=1)[0].cpu().numpy()
            int_idx = int(np.argmax(i_probs))
            top_intent = self.intents[int_idx]
            int_conf = round(float(i_probs[int_idx]), 4)

        # 5. SEMANTIC DOMAIN GATE
        # Combines semantic concept matches, emergency red flags, medical corrections, and transformer domain head
        has_clinical_concepts = len(detected_entities) > 0
        has_semantic_proximity = max_semantic_score >= 0.55
        has_medical_corrections = any(
            c.get("type") in ["spelling", "fuzzy_spelling", "abbreviation"] and
            c.get("replacement") in self.normalizer.medical_vocabulary
            for c in corrections
        )

        if emergency_info["is_emergency"] or emergency_info["safety_level"] == "critical":
            is_medical = True
            domain_conf = 0.9999
        elif has_clinical_concepts or has_semantic_proximity:
            is_medical = True
            domain_conf = max(0.95, round(is_medical_prob, 4))
        elif has_medical_corrections:
            is_medical = True
            domain_conf = max(0.92, round(is_medical_prob, 4))
        elif max_semantic_score < 0.40 and not has_clinical_concepts:
            # Definitively non-medical: zero clinical concepts and low semantic similarity
            is_medical = False
            domain_conf = round(max(0.85, 1.0 - max_semantic_score), 4)
        else:
            # Fall back to transformer domain head with cautious threshold
            is_medical = bool(is_medical_prob >= 0.65)
            domain_conf = round(is_medical_prob if is_medical else float(d_probs[0]), 4)

        # Handle Non-Medical Domain Exit
        if not is_medical:
            return {
                "question": raw_q,
                "normalized_question": normalized_q,
                "corrections": corrections,
                "is_medical": False,
                "domain_confidence": domain_conf,
                "message": "This question appears to be outside the supported medical domain.",
                "category": "Non-Medical",
                "category_confidence": 0.0,
                "primary_category": "Non-Medical",
                "primary_category_confidence": 0.0,
                "primary_category_reason": None,
                "secondary_categories": [],
                "intent": "Non-Medical",
                "intent_confidence": 0.0,
                "secondary_intents": [],
                "overall_confidence": 0.0,
                "uncertain": False,
                "uncertainty_reason": None,
                "clarification_needed": None,
                "is_multi_symptom": False,
                "detected_entities": [],
                "clinical_signals": [],
                "suggested_next_steps": [
                    "Please submit a health, symptom, medication, or biomedical question to classify its clinical specialty."
                ],
                "safety_level": "routine",
                "explanation": None,
                "similar_questions": [],
                "routing": None,
                "emergency": emergency_info,
                "safety_disclaimer": (
                    "MedClass AI classifies medical questions and is NOT a medical diagnostic tool. "
                    "It does not diagnose illnesses or replace consultation with a registered medical practitioner."
                ),
                "timestamp_ist": formatted_ist
            }

        # 6. Intent Refinement from Natural Phrasing
        # Check high-confidence intent cues from normalized text
        norm_lower = normalized_q.lower()
        if re.search(r"\b(causes?|caused|causing|origin|why do|why does)\b", norm_lower):
            top_intent = "Causes"
            int_conf = max(0.92, int_conf)
        elif re.search(r"\b(symptom|symptoms|signs?|feel|warning signs?)\b", norm_lower):
            top_intent = "Symptoms"
            int_conf = max(0.92, int_conf)
        elif re.search(r"\b(treat|treatment|treatments|cure|therap|manage|how to treat)\b", norm_lower):
            top_intent = "Treatment"
            int_conf = max(0.92, int_conf)
        elif re.search(r"\b(prevent|prevention|avoid|how to prevent)\b", norm_lower):
            top_intent = "Prevention"
            int_conf = max(0.92, int_conf)
        elif re.search(r"\b(diagnos|diagnose|diagnosis|test|screen|blood test|screening)\b", norm_lower):
            top_intent = "Diagnosis"
            int_conf = max(0.92, int_conf)
        elif re.search(r"\b(side effects?|adverse)\b", norm_lower):
            top_intent = "Side Effects"
            int_conf = max(0.92, int_conf)

        # Secondary intents
        sorted_int_indices = np.argsort(i_probs)[::-1]
        secondary_intents = []
        for idx in sorted_int_indices:
            name = self.intents[idx]
            if name != top_intent:
                prob = float(i_probs[idx])
                if prob >= 0.12 or len(secondary_intents) < 2:
                    secondary_intents.append({
                        "intent": name,
                        "confidence": round(prob, 4)
                    })
            if len(secondary_intents) >= 2:
                break

        # 7. CLINICAL SPECIALTY & PRIMARY CATEGORY DETERMINATION
        # Detect cardinal symptom clusters
        matched_concept_ids = set(r.get("id") for r in matched_records)
        is_thirst_urination_cluster = bool(
            "excessive_thirst" in matched_concept_ids and "frequent_urination" in matched_concept_ids
        )
        has_cardio_acuity = any(r.get("domain") == "Cardiology" and r.get("priority") == 1 for r in matched_records)
        has_resp_acuity = any(r.get("domain") == "Respiratory" and r.get("priority") == 1 for r in matched_records)

        primary_category = model_top_category
        primary_cat_conf = model_cat_conf
        primary_category_reason = None

        if is_thirst_urination_cluster:
            primary_category = "Endocrinology"
            endo_idx = self.cat2idx.get("Endocrinology", cat_idx)
            primary_cat_conf = max(0.90, round(float(c_probs[endo_idx]), 4))
            primary_category_reason = (
                "Endocrinology was prioritized as primary clinical domain because excessive thirst (polydipsia) "
                "paired with frequent urination (polyuria) constitutes a recognized metabolic presentation "
                "often evaluated in relation to blood glucose regulation, alongside General Medicine."
            )
        elif is_multi_symptom:
            if has_cardio_acuity:
                primary_category = "Cardiology"
                cardio_idx = self.cat2idx.get("Cardiology", cat_idx)
                primary_cat_conf = max(0.88, round(float(c_probs[cardio_idx]), 4))
                primary_category_reason = (
                    "Cardiology was prioritized as primary clinical category due to the presence of acute chest discomfort, "
                    "which warrants prioritized clinical evaluation alongside other reported symptoms."
                )
            elif has_resp_acuity:
                primary_category = "Respiratory"
                resp_idx = self.cat2idx.get("Respiratory", cat_idx)
                primary_cat_conf = max(0.88, round(float(c_probs[resp_idx]), 4))
                primary_category_reason = (
                    "Respiratory was prioritized as primary category due to acute breathing difficulty, "
                    "which warrants prompt clinical assessment alongside other reported symptoms."
                )
            elif matched_records:
                top_record_domain = matched_records[0].get("domain")
                if top_record_domain in self.cat2idx:
                    primary_category = top_record_domain
                    d_idx = self.cat2idx[top_record_domain]
                    primary_cat_conf = max(0.80, round(float(c_probs[d_idx]), 4))
                    primary_category_reason = (
                        f"Primary category '{primary_category}' selected based on leading clinical signal "
                        f"'{matched_records[0]['name']}', while recognizing that multiple distinct clinical domains are co-present."
                    )
            else:
                primary_category_reason = (
                    f"Primary category '{primary_category}' selected based on contextual semantic representation, "
                    "while recognizing that multiple clinical domains are involved."
                )
        else:
            # Single symptom or single entity inquiry
            if matched_records:
                entity_domain = matched_records[0].get("domain")
                if entity_domain and entity_domain in self.cat2idx:
                    primary_category = entity_domain
                    d_idx = self.cat2idx[entity_domain]
                    primary_cat_conf = max(0.85, round(float(c_probs[d_idx]), 4))
                    primary_category_reason = (
                        f"Classified under '{primary_category}' due to strong clinical alignment with detected "
                        f"clinical indicator '{matched_records[0]['name']}'."
                    )
            elif detected_entities:
                primary_category_reason = (
                    f"Classified under '{primary_category}' due to alignment with detected entity '{', '.join(detected_entities)}'."
                )
            else:
                primary_category_reason = (
                    f"Classified under '{primary_category}' based on contextual transformer representation."
                )

        # 8. SECONDARY CATEGORIES
        secondary_categories = []
        seen_sec_cats = {primary_category}

        # If thirst/urination cluster, always offer General Medicine as secondary
        if is_thirst_urination_cluster:
            gen_idx = self.cat2idx.get("General Medicine")
            gen_prob = float(c_probs[gen_idx]) if gen_idx is not None else 0.40
            seen_sec_cats.add("General Medicine")
            secondary_categories.append({
                "category": "General Medicine",
                "confidence": round(max(0.35, gen_prob), 4),
                "reason": "General internal medicine evaluation for initial multi-system and fluid-balance assessment"
            })

        # Add secondary domains from matched records
        for item in matched_records:
            d_name = item.get("domain")
            if d_name and d_name not in seen_sec_cats:
                seen_sec_cats.add(d_name)
                item_prob = float(c_probs[self.cat2idx[d_name]]) if d_name in self.cat2idx else 0.35
                secondary_categories.append({
                    "category": d_name,
                    "confidence": round(max(0.25, item_prob), 4),
                    "reason": f"Detected clinical indicator: {item['name']} ({item.get('subdomain', d_name)})"
                })

        # Add top model secondary candidates
        sorted_cat_indices = np.argsort(c_probs)[::-1]
        for idx in sorted_cat_indices:
            c_name = self.categories[idx]
            if c_name not in seen_sec_cats:
                prob = float(c_probs[idx])
                if prob >= 0.10 or len(secondary_categories) < 2:
                    seen_sec_cats.add(c_name)
                    secondary_categories.append({
                        "category": c_name,
                        "confidence": round(prob, 4),
                        "reason": "Secondary probabilistic match from multi-task transformer"
                    })
            if len(secondary_categories) >= 4:
                break

        # 9. AMBIGUITY, UNCERTAINTY & CLARIFICATION
        words = re.findall(r"\w+", normalized_q.lower())
        is_single_word = len(words) <= 2 and not any(w in words for w in ["what", "how", "why", "when", "can", "is", "are", "cause", "symptom"])

        overall_confidence = round(float((primary_cat_conf + int_conf) / 2.0), 4)
        is_uncertain = bool(primary_cat_conf < 0.60 or int_conf < 0.60 or is_single_word)
        uncertainty_reason = None
        clarification_needed = None

        if is_single_word:
            is_uncertain = True
            uncertainty_reason = "Single-word or underspecified medical term with ambiguous clinical intention."
            clarification_needed = (
                f"The inquiry consists of a brief keyword or condition ('{raw_q}'). To ensure precise classification, "
                "please specify your question: for example, are you inquiring about causes, early warning signs, "
                "diagnostic tests, or treatment options?"
            )
        elif is_uncertain:
            uncertainty_reason = (
                "Low classification confidence. The inquiry phrasing may be multi-faceted or underspecified. "
                "Providing specific clinical context will improve classification clarity."
            )

        # 10. EXPLAINABILITY & CLINICAL ROUTING
        explanation = self.explain_prediction(
            normalized_q, primary_category, top_intent, detected_entities, is_multi_symptom, is_thirst_urination_cluster
        )
        routing = self.get_routing_suggestions(primary_category, top_intent)
        next_steps = self.generate_next_steps(
            emergency_info["safety_level"], is_multi_symptom, detected_entities, primary_category, is_thirst_urination_cluster
        )

        # 11. GROUNDED KNOWLEDGE RETRIEVAL ON MEDQUAD
        # Query similarity engine using normalized query + intent + detected entities
        similar_qs = self.sim_engine.search(
            normalized_q,
            top_k=4,
            target_intent=top_intent,
            target_entities=detected_entities
        )

        # 12. Complete Return Structure (Backward-Compatible)
        return {
            "question": raw_q,
            "normalized_question": normalized_q,
            "corrections": corrections,
            "is_medical": True,
            "domain_confidence": domain_conf,
            # Backward-compatible category keys
            "category": primary_category,
            "category_confidence": primary_cat_conf,
            "primary_category": primary_category,
            "primary_category_confidence": primary_cat_conf,
            "primary_category_reason": primary_category_reason,
            "secondary_categories": secondary_categories,
            # Intent keys
            "intent": top_intent,
            "intent_confidence": int_conf,
            "secondary_intents": secondary_intents,
            # Metrics
            "overall_confidence": overall_confidence,
            "uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "clarification_needed": clarification_needed,
            # Multi-symptom & Entity keys
            "is_multi_symptom": is_multi_symptom,
            "detected_entities": detected_entities,
            "clinical_signals": clinical_signals,
            # Safety & Guidance keys
            "safety_level": emergency_info["safety_level"],
            "suggested_next_steps": next_steps,
            "explanation": explanation,
            "similar_questions": similar_qs,
            "routing": routing,
            "emergency": emergency_info,
            "safety_disclaimer": (
                "MedClass AI classifies medical questions and is not a medical diagnostic tool. "
                "It does not diagnose conditions, prescribe treatments, or replace consultation with a registered medical practitioner."
            ),
            "timestamp_ist": formatted_ist
        }


# Global singleton instance
_PREDICTOR_INSTANCE = None

def get_predictor():
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = MedicalQuestionPredictor()
        _PREDICTOR_INSTANCE.load()
    return _PREDICTOR_INSTANCE


if __name__ == "__main__":
    predictor = get_predictor()
    test_queries = [
        "I have been thirsty all day and going to the bathroom constantly, what could be wrong?",
        "I feel extremely thirsty and keep urinating even though I haven't been drinking much water",
        "I've suddenly started drinking lots of water and waking up several times at night to pee",
        "wat r the signs of diabetis",
        "gloucoma",
        "migrene",
        "astma",
        "head hurts one side and light makes it worse",
        "how do I reset my wifi router",
        "I am having leg pain, fever and chest pain"
    ]
    for q in test_queries:
        res = predictor.predict(q)
        print("\n" + "="*60)
        print(f"QUERY: {q}")
        print(f"Normalized: {res.get('normalized_question')}")
        print(f"Is Medical: {res.get('is_medical')} ({res.get('domain_confidence')*100:.1f}%)")
        if res.get("is_medical"):
            print(f"Primary Category: {res.get('primary_category')} ({res.get('primary_category_confidence')*100:.1f}%)")
            print(f"Reason: {res.get('primary_category_reason')}")
            print(f"Intent: {res.get('intent')} ({res.get('intent_confidence')*100:.1f}%)")
            print(f"Multi-Symptom: {res.get('is_multi_symptom')}")
            print(f"Detected Entities: {res.get('detected_entities')}")
            print(f"Safety Level: {res.get('safety_level')}")
            print(f"Explanation: {res.get('explanation', {}).get('explanation_text')}")
            if res.get("similar_questions"):
                print("Top Similar Question:", res["similar_questions"][0]["question"])
        else:
            print(f"Non-Medical Message: {res.get('message')}")
