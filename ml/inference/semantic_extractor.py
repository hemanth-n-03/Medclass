"""
MedClass AI Semantic Clinical Concept & Entity Extraction Engine
Uses Sentence-Transformers (all-MiniLM-L6-v2) to map colloquial, informal,
and symptom-based user descriptions to canonical clinical concepts and specialties.
Strictly non-diagnostic: maps symptoms to clinical domains without medical diagnosis.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CLINICAL_CONCEPTS_REGISTRY = [
    # Endocrinology
    {
        "id": "excessive_thirst",
        "name": "Excessive thirst (Polydipsia)",
        "domain": "Endocrinology",
        "subdomain": "Endocrine / Metabolic",
        "priority": 3,
        "signal": "Excessive thirst (polydipsia) is an osmotic symptom commonly evaluated in relation to blood glucose regulation, fluid balance, or endocrine function.",
        "patterns": [
            r"thirst", r"thirsty", r"drinking (lots of|so much|a lot of|nonstop) water",
            r"dry mouth and (keep )?drinking", r"polydipsia", r"mouth is always dry"
        ],
        "exemplars": [
            "excessive thirst", "drinking water nonstop", "always thirsty", "extremely thirsty",
            "mouth is always dry and I keep drinking", "unquenchable thirst", "feeling thirsty all day",
            "drinking lots of water constantly", "parched mouth and drinking fluids constantly"
        ]
    },
    {
        "id": "frequent_urination",
        "name": "Frequent urination (Polyuria)",
        "domain": "Endocrinology",
        "subdomain": "Endocrine / Metabolic",
        "priority": 3,
        "signal": "Frequent urination (polyuria), including nocturia, indicates increased osmotic diuresis or lower urinary tract irritation.",
        "patterns": [
            r"urinat(e|ing|ion)", r"peeing", r"pee a lot", r"going to the bathroom constantly",
            r"running to the toilet", r"polyuria", r"waking up .* to pee", r"pee every few minutes"
        ],
        "exemplars": [
            "frequent urination", "peeing all the time", "going to the bathroom constantly",
            "running to the toilet a lot", "need to pee every few minutes", "urinating frequently",
            "waking up several times at night to pee", "urinating constantly", "pee a lot throughout the day"
        ]
    },
    {
        "id": "diabetes_metabolic",
        "name": "Diabetes / Blood sugar regulation",
        "domain": "Endocrinology",
        "subdomain": "Endocrine / Metabolic",
        "priority": 3,
        "signal": "Inquiries concerning blood glucose regulation relate to pancreatic endocrine function and insulin metabolism.",
        "patterns": [
            r"diabetes", r"diabetic", r"blood sugar", r"hyperglycemia", r"hypoglycemia",
            r"glucose level", r"insulin", r"a1c"
        ],
        "exemplars": [
            "signs of diabetes", "high blood sugar levels", "low blood glucose symptoms",
            "insulin resistance test", "early symptoms of type 2 diabetes", "glycemic control"
        ]
    },
    {
        "id": "thyroid_disorder",
        "name": "Thyroid disorder",
        "domain": "Endocrinology",
        "subdomain": "Endocrine",
        "priority": 3,
        "signal": "Thyroid indicators reflect metabolic hormone synthesis by the thyroid gland.",
        "patterns": [
            r"thyroid", r"hypothyroidism", r"hyperthyroidism", r"goiter", r"hashimoto", r"tsh"
        ],
        "exemplars": [
            "thyroid hormone imbalance", "underactive thyroid symptoms", "overactive thyroid signs",
            "goiter swelling in neck", "hypothyroidism fatigue and weight gain"
        ]
    },

    # Cardiology
    {
        "id": "chest_pain_acute",
        "name": "Chest pain",
        "domain": "Cardiology",
        "subdomain": "Cardiovascular / Acute",
        "priority": 1,
        "signal": "Chest discomfort or pressure can be associated with acute cardiovascular conditions requiring prompt medical assessment.",
        "patterns": [
            r"chest pain", r"chest tightness", r"chest pressure", r"angina", r"heavy chest",
            r"heart pain", r"elephant on my chest"
        ],
        "exemplars": [
            "chest pain", "crushing chest pressure", "tightness in chest", "heavy chest feeling",
            "pain in heart", "angina pectoris", "squeezing chest discomfort", "sharp stabbing chest pain"
        ]
    },
    {
        "id": "radiating_pain",
        "name": "Radiating pain (arm/jaw/neck)",
        "domain": "Cardiology",
        "subdomain": "Cardiovascular / Acute",
        "priority": 1,
        "signal": "Pain radiating into the arm, jaw, neck, or shoulder is a recognized clinical indicator warranting urgent evaluation.",
        "patterns": [
            r"radiating (to|down|into) (my )?(left )?(arm|jaw|neck|shoulder|back)",
            r"pain in (my )?(left )?arm and chest"
        ],
        "exemplars": [
            "pain radiating to left arm", "pain shooting down arm", "jaw pain and neck pain with chest ache",
            "pain spreading to shoulder and back", "left shoulder and arm numbness with chest pain"
        ]
    },
    {
        "id": "palpitations",
        "name": "Heart palpitations / Arrhythmia",
        "domain": "Cardiology",
        "subdomain": "Cardiovascular / Electrophysiology",
        "priority": 2,
        "signal": "Heart palpitations indicate conscious awareness of rapid, irregular, or forceful cardiac contractions.",
        "patterns": [
            r"palpitation", r"irregular heartbeat", r"racing heart", r"fluttering (in|of) (my )?chest",
            r"skipping beats", r"rapid pulse", r"pounding heart"
        ],
        "exemplars": [
            "heart feels like it is skipping beats when I lie down", "racing heart rate",
            "heart fluttering in chest", "irregular heartbeat palpitations", "rapid pounding pulse at rest",
            "skipped heartbeats sensation"
        ]
    },
    {
        "id": "hypertension",
        "name": "Hypertension / High blood pressure",
        "domain": "Cardiology",
        "subdomain": "Cardiovascular / Chronic",
        "priority": 3,
        "signal": "Elevated blood pressure is a major cardiovascular parameter requiring systematic clinical monitoring.",
        "patterns": [
            r"high blood pressure", r"hypertension", r"elevated bp", r"hypertensive", r"blood pressure"
        ],
        "exemplars": [
            "high blood pressure readings", "symptoms of hypertension", "elevated blood pressure crisis",
            "how to lower high bp", "hypertensive heart strain"
        ]
    },

    # Respiratory
    {
        "id": "shortness_of_breath",
        "name": "Shortness of breath / Dyspnea",
        "domain": "Respiratory",
        "subdomain": "Pulmonary / Acute",
        "priority": 1,
        "signal": "Shortness of breath indicates impaired gas exchange or increased work of breathing requiring clinical scrutiny.",
        "patterns": [
            r"difficulty breathing", r"shortness of breath", r"breathlessness", r"can'?t breathe",
            r"dyspnea", r"struggling to breathe", r"can'?t catch my breath"
        ],
        "exemplars": [
            "shortness of breath", "can't catch my breath after walking up stairs", "struggling to breathe",
            "breathlessness with exertion", "difficulty taking a deep breath", "hard to breathe when lying down",
            "gasping for air"
        ]
    },
    {
        "id": "cough_wheezing",
        "name": "Cough / Wheezing / Asthma",
        "domain": "Respiratory",
        "subdomain": "Pulmonary",
        "priority": 2,
        "signal": "Coughing and wheezing reflect airway hyper-reactivity, bronchospasm, or respiratory tract inflammation.",
        "patterns": [
            r"\bcough\b", r"coughing", r"wheezing", r"phlegm", r"bronchitis", r"asthma",
            r"chest congestion", r"mucus"
        ],
        "exemplars": [
            "asthma attack wheezing", "persistent dry cough", "coughing up phlegm and mucus",
            "wheezing sound when exhaling", "bronchitis chest congestion", "asthma inhaler"
        ]
    },

    # Neurology
    {
        "id": "headache_migraine",
        "name": "Headache / Migraine",
        "domain": "Neurology",
        "subdomain": "Neurological",
        "priority": 2,
        "signal": "Headache presentations range from primary vascular/tension headaches to secondary neurological states.",
        "patterns": [
            r"headache", r"migraine", r"throbbing head", r"head pain", r"tension headache",
            r"head hurts"
        ],
        "exemplars": [
            "head hurts one side and light makes it worse", "throbbing migraine headache",
            "severe headache on left side", "pounding head pain with nausea", "tension headache temples"
        ]
    },
    {
        "id": "photophobia_aura",
        "name": "Photophobia / Sensory sensitivity",
        "domain": "Neurology",
        "subdomain": "Neurological",
        "priority": 3,
        "signal": "Light sensitivity (photophobia) and visual auras frequently accompany neurological headache pathways.",
        "patterns": [
            r"light makes (it|headache) worse", r"sensitive to light", r"photophobia",
            r"bright light hurts", r"visual aura", r"flashing lights in eyes"
        ],
        "exemplars": [
            "light makes my headache worse", "sensitive to bright lights and loud sounds",
            "photophobia light sensitivity", "visual aura seeing zigzag patterns"
        ]
    },
    {
        "id": "focal_neurological",
        "name": "Focal neurological deficit (FAST)",
        "domain": "Neurology",
        "subdomain": "Neurological / Acute",
        "priority": 1,
        "signal": "Focal neurological deficits such as facial drooping, slurred speech, or unilateral weakness are critical acute stroke signs.",
        "patterns": [
            r"slurred speech", r"facial drooping", r"sudden numbness", r"sudden weakness",
            r"loss of balance", r"one-sided weakness", r"face drooping"
        ],
        "exemplars": [
            "slurred speech unable to speak clearly", "face drooping on one side",
            "sudden weakness in arm and leg", "sudden numbness in half of body", "loss of balance and coordination"
        ]
    },
    {
        "id": "dizziness_vertigo",
        "name": "Dizziness / Vertigo",
        "domain": "Neurology",
        "subdomain": "Neurological",
        "priority": 3,
        "signal": "Dizziness suggests vestibular disturbance, central neurological etiology, or hemodynamic fluctuation.",
        "patterns": [
            r"dizziness", r"vertigo", r"lightheadedness", r"feeling faint", r"unsteadiness", r"room spinning"
        ],
        "exemplars": [
            "room is spinning vertigo", "feeling dizzy and lightheaded", "unsteadiness when standing up",
            "loss of balance and dizziness", "feeling faint and woozy"
        ]
    },

    # Infectious Disease
    {
        "id": "fever_systemic",
        "name": "Fever",
        "domain": "Infectious Disease",
        "subdomain": "Infectious / Systemic",
        "priority": 2,
        "signal": "Fever reflects an active systemic physiological response often associated with infectious or inflammatory processes.",
        "patterns": [
            r"\bfever\b", r"high temperature", r"pyrexia", r"chills", r"sweating", r"shivering",
            r"feverish", r"high temp"
        ],
        "exemplars": [
            "kid has high temp and won't eat anything", "running a high fever with chills",
            "body temperature over 102 degrees", "shivering and sweating with fever",
            "pyrexia and viral illness"
        ]
    },
    {
        "id": "fatigue_malaise",
        "name": "Fatigue / Malaise",
        "domain": "Infectious Disease",
        "subdomain": "Infectious / Systemic",
        "priority": 4,
        "signal": "Fatigue and generalized malaise indicate constitutional or systemic immune activation.",
        "patterns": [
            r"fatigue", r"exhaustion", r"tiredness", r"malaise", r"body ache", r"generalized weakness"
        ],
        "exemplars": [
            "extreme fatigue and tiredness", "generalized body aches and exhaustion",
            "feeling weak all over", "constant fatigue and lack of energy"
        ]
    },

    # Orthopedics & Rheumatology
    {
        "id": "leg_pain",
        "name": "Leg pain",
        "domain": "Orthopedics & Rheumatology",
        "subdomain": "Musculoskeletal",
        "priority": 3,
        "signal": "Leg pain commonly involves musculoskeletal structures, though vascular or systemic contributors should be considered in complex presentations.",
        "patterns": [
            r"leg pain", r"pain in (my )?legs?", r"calf pain", r"thigh pain", r"shin splints", r"cramp in leg"
        ],
        "exemplars": [
            "leg pain when walking", "calf ache and muscle pain", "pain in both legs",
            "thigh muscle soreness", "shin splints and leg stiffness"
        ]
    },
    {
        "id": "joint_pain_arthritis",
        "name": "Joint pain / Arthritis",
        "domain": "Orthopedics & Rheumatology",
        "subdomain": "Musculoskeletal",
        "priority": 3,
        "signal": "Joint discomfort and swelling are characteristic of musculoskeletal or rheumatologic inflammatory pathways.",
        "patterns": [
            r"joint pain", r"knee pain", r"swollen joint", r"arthritis", r"stiff joints?",
            r"hip pain", r"shoulder pain"
        ],
        "exemplars": [
            "stiff painful knee joints", "swollen joint in fingers arthritis", "rheumatoid arthritis joint pain",
            "shoulder joint ache and limited mobility", "hip pain when moving"
        ]
    },
    {
        "id": "back_pain",
        "name": "Back pain / Spine discomfort",
        "domain": "Orthopedics & Rheumatology",
        "subdomain": "Musculoskeletal",
        "priority": 3,
        "signal": "Back discomfort is predominantly musculoskeletal, requiring observation for any red-flag neurological deficits.",
        "patterns": [
            r"back pain", r"lower back pain", r"spine pain", r"lumbago", r"sciatica"
        ],
        "exemplars": [
            "severe lower back pain", "spine stiffness and ache", "sciatica nerve shooting down leg",
            "lumbar back strain", "pain in upper back between shoulder blades"
        ]
    },

    # Gastroenterology
    {
        "id": "abdominal_pain",
        "name": "Abdominal pain",
        "domain": "Gastroenterology",
        "subdomain": "Gastrointestinal",
        "priority": 2,
        "signal": "Abdominal pain encompasses visceral, mucosal, and digestive inflammatory pathways.",
        "patterns": [
            r"abdominal pain", r"stomach pain", r"belly ache", r"stomach cramps?", r"gut pain"
        ],
        "exemplars": [
            "cramping stomach pain", "sharp abdominal pain after eating", "belly ache and cramps",
            "severe pain in lower abdomen", "gut discomfort"
        ]
    },
    {
        "id": "digestive_gi_symptoms",
        "name": "Digestive symptoms (nausea/reflux/diarrhea)",
        "domain": "Gastroenterology",
        "subdomain": "Gastrointestinal",
        "priority": 3,
        "signal": "Upper and lower gastrointestinal symptoms reflect motility disturbance or mucosal irritation.",
        "patterns": [
            r"nausea", r"vomiting", r"diarrhea", r"acid reflux", r"heartburn", r"gerd", r"indigestion"
        ],
        "exemplars": [
            "nausea and throwing up", "severe diarrhea and stomach upset", "acid reflux and burning heartburn",
            "indigestion and bloating after meals", "chronic vomiting and dehydration"
        ]
    },

    # Dermatology
    {
        "id": "skin_rash_pruritus",
        "name": "Skin rash / Pruritus",
        "domain": "Dermatology",
        "subdomain": "Dermatological",
        "priority": 4,
        "signal": "Cutaneous presentations indicate epidermal barrier breakdown, allergic response, or immune-mediated dermatosis.",
        "patterns": [
            r"rash", r"skin lesion", r"itching", r"eczema", r"hives", r"blisters?",
            r"psoriasis", r"dermatitis", r"red patches"
        ],
        "exemplars": [
            "my skin has red patches that itch like crazy", "itchy rash on arms and legs",
            "eczema flare up dry itchy skin", "hives and allergic skin reaction", "red inflamed itchy bumps on skin"
        ]
    },

    # Medication & Pharmacology
    {
        "id": "medication_pharmacology",
        "name": "Medication & Side effects",
        "domain": "Medication & Pharmacology",
        "subdomain": "Pharmacological",
        "priority": 3,
        "signal": "Pharmacological questions center on drug mechanisms, safety profiles, interactions, and adverse events.",
        "patterns": [
            r"side effects?", r"adverse (reaction|effect)", r"drug interaction", r"antibiotics?",
            r"medication", r"dosage", r"prescription"
        ],
        "exemplars": [
            "side effects of prescription medication", "drug interaction between antibiotics",
            "proper dosage for medicine", "adverse reaction to pills", "is this medication safe to take"
        ]
    },

    # Mental Health
    {
        "id": "mental_health",
        "name": "Mental health / Psychological distress",
        "domain": "Mental Health",
        "subdomain": "Psychological",
        "priority": 2,
        "signal": "Psychological symptoms relate to cognitive-emotional functioning and neurochemical affective balance.",
        "patterns": [
            r"depression", r"anxiety", r"panic attacks?", r"insomnia", r"suicid", r"self-harm",
            r"mental breakdown"
        ],
        "exemplars": [
            "severe depression and feeling hopeless", "panic attacks and extreme anxiety",
            "chronic insomnia unable to sleep", "mental health crisis anxiety disorder"
        ]
    },

    # General Medicine / Ophthalmology
    {
        "id": "ocular_vision",
        "name": "Vision / Ocular changes",
        "domain": "General Medicine",
        "subdomain": "Ophthalmology",
        "priority": 3,
        "signal": "Visual disturbances and ocular symptoms warrant evaluation for corneal, retinal, or intraocular pressure pathology.",
        "patterns": [
            r"glaucoma", r"blurry vision", r"vision loss", r"eye pain", r"seeing halos", r"cataract"
        ],
        "exemplars": [
            "glaucoma eye pressure symptoms", "blurry vision in one eye", "halos around lights",
            "gradual loss of peripheral vision", "severe eye ache and redness"
        ]
    }
]


class SemanticClinicalExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder: Optional[SentenceTransformer] = None
        self.is_loaded = False
        
        # Precomputed exemplar vectors and mapping
        self.concept_definitions = CLINICAL_CONCEPTS_REGISTRY
        self.exemplar_texts: List[str] = []
        self.exemplar_to_concept: List[Dict[str, Any]] = []
        self.exemplar_embeddings: Optional[np.ndarray] = None

    def load(self):
        if self.is_loaded:
            return

        # Load sentence transformer
        self.encoder = SentenceTransformer(self.model_name)

        # Prepare all exemplar texts
        self.exemplar_texts = []
        self.exemplar_to_concept = []
        for concept in self.concept_definitions:
            for ex in concept["exemplars"]:
                self.exemplar_texts.append(ex)
                self.exemplar_to_concept.append(concept)

        # Compute normalized exemplar embeddings
        self.exemplar_embeddings = self.encoder.encode(
            self.exemplar_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        self.is_loaded = True

    def extract(
        self,
        text: str,
        normalized_text: Optional[str] = None,
        similarity_threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Extracts clinical entities, signals, and matched concept records
        using hybrid regex + dense sentence embeddings.
        """
        if not self.is_loaded:
            self.load()

        texts_to_check = [text]
        if normalized_text and normalized_text.lower() != text.lower():
            texts_to_check.append(normalized_text)

        matched_concepts: Dict[str, Dict[str, Any]] = {}
        max_similarity_overall = 0.0

        # 1. Fast Regex Pattern Matching Path
        for t in texts_to_check:
            t_lower = t.lower()
            for concept in self.concept_definitions:
                c_id = concept["id"]
                if c_id not in matched_concepts:
                    for pat in concept["patterns"]:
                        if re.search(pat, t_lower):
                            matched_concepts[c_id] = {
                                "concept": concept,
                                "match_type": "pattern",
                                "similarity": 1.0
                            }
                            max_similarity_overall = 1.0
                            break

        # 2. Dense Semantic Cosine Similarity Path
        # Encode user query texts
        q_embeddings = self.encoder.encode(
            texts_to_check,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Cosine similarity against all exemplars: shape (n_queries, n_exemplars)
        sim_matrix = np.dot(q_embeddings, self.exemplar_embeddings.T)

        for q_idx in range(len(texts_to_check)):
            scores = sim_matrix[q_idx]
            top_score_for_query = float(np.max(scores)) if len(scores) > 0 else 0.0
            if top_score_for_query > max_similarity_overall:
                max_similarity_overall = top_score_for_query

            # Group scores by concept ID
            concept_max_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
            for ex_idx, score in enumerate(scores):
                concept = self.exemplar_to_concept[ex_idx]
                c_id = concept["id"]
                val = float(score)
                if c_id not in concept_max_scores or val > concept_max_scores[c_id][0]:
                    concept_max_scores[c_id] = (val, concept)

            # Check threshold
            for c_id, (top_val, concept) in concept_max_scores.items():
                if top_val >= similarity_threshold:
                    if c_id not in matched_concepts or top_val > matched_concepts[c_id]["similarity"]:
                        matched_concepts[c_id] = {
                            "concept": concept,
                            "match_type": "semantic",
                            "similarity": round(top_val, 4)
                        }

        # 3. Assemble results
        detected_entities = []
        clinical_signals = []
        matched_records = []

        # Sort by priority then similarity
        sorted_matches = sorted(
            matched_concepts.values(),
            key=lambda x: (x["concept"]["priority"], -x["similarity"])
        )

        for m in sorted_matches:
            c = m["concept"]
            detected_entities.append(c["name"])
            clinical_signals.append({
                "entity": c["name"],
                "domain": c["subdomain"],
                "signal": c["signal"],
                "confidence": m["similarity"]
            })
            matched_records.append(c)

        unique_domains = list(set(r["domain"] for r in matched_records))
        is_multi_symptom = len(unique_domains) >= 2 or len(detected_entities) >= 2

        return {
            "detected_entities": detected_entities,
            "clinical_signals": clinical_signals,
            "matched_records": matched_records,
            "is_multi_symptom": is_multi_symptom,
            "max_semantic_score": max_similarity_overall
        }


# Singleton instance
_EXTRACTOR_INSTANCE = None

def get_semantic_extractor() -> SemanticClinicalExtractor:
    global _EXTRACTOR_INSTANCE
    if _EXTRACTOR_INSTANCE is None:
        _EXTRACTOR_INSTANCE = SemanticClinicalExtractor()
        _EXTRACTOR_INSTANCE.load()
    return _EXTRACTOR_INSTANCE
