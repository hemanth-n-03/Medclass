"""
MedClass AI Query Normalization & Preprocessing Pipeline
Handles:
1. Common medical abbreviations and acronyms
2. Spelling correction for medical conditions and clinical terms
3. Conversational slang and contraction expansion
4. Conversational filler stabilization
Preserves original query for auditing while generating normalized representations for inference.
"""

import os
import re
from typing import Dict, Any, List, Tuple, Set

# Known high-frequency medical misspellings
COMMON_MEDICAL_TYPOS = {
    "diabetis": "diabetes",
    "diabates": "diabetes",
    "diabeetus": "diabetes",
    "gloucoma": "glaucoma",
    "glaucom": "glaucoma",
    "glacoma": "glaucoma",
    "migrene": "migraine",
    "migrane": "migraine",
    "mygraine": "migraine",
    "astma": "asthma",
    "athsma": "asthma",
    "asthm": "asthma",
    "hypertention": "hypertension",
    "hipertension": "hypertension",
    "hypertesion": "hypertension",
    "alzhiemer": "alzheimer",
    "alzhiemers": "alzheimers",
    "alzhiemer's": "alzheimer's",
    "alzheimrs": "alzheimers",
    "bronchitus": "bronchitis",
    "broncitus": "bronchitis",
    "arthritus": "arthritis",
    "arthiritis": "arthritis",
    "colestrol": "cholesterol",
    "choleserol": "cholesterol",
    "cholestrol": "cholesterol",
    "pneumona": "pneumonia",
    "pnuemonia": "pneumonia",
    "pnemonia": "pneumonia",
    "diarreah": "diarrhea",
    "diarhea": "diarrhea",
    "diarea": "diarrhea",
    "nausia": "nausea",
    "nauseau": "nausea",
    "headace": "headache",
    "hedache": "headache",
    "stomac": "stomach",
    "stomache": "stomach",
    "infetion": "infection",
    "infaction": "infection",
    "leukimia": "leukemia",
    "lukemia": "leukemia",
    "psoraisis": "psoriasis",
    "eczyma": "eczema",
    "excema": "eczema",
    "alergic": "allergic",
    "alergy": "allergy",
    "insomna": "insomnia",
    "epilepsi": "epilepsy",
    "arthroplasty": "arthroplasty",
    "arrithmia": "arrhythmia",
    "arythmia": "arrhythmia",
    "anurism": "aneurysm",
    "aneurism": "aneurysm",
}

# Slang, abbreviations, and informal shorthand
COMMON_ABBREVIATIONS = {
    "wat": "what",
    "wats": "what is",
    "wat's": "what is",
    "r": "are",
    "u": "you",
    "ur": "your",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "doc": "doctor",
    "dr": "doctor",
    "dr.": "doctor",
    "bp": "blood pressure",
    "b/p": "blood pressure",
    "hr": "heart rate",
    "temp": "temperature",
    "rx": "prescription medication",
    "otc": "over the counter medication",
    "sob": "shortness of breath",
    "uti": "urinary tract infection",
    "gerd": "acid reflux",
    "dvt": "deep vein thrombosis",
    "ent": "ear nose throat",
    "gi": "gastrointestinal",
    "w/": "with",
    "w/o": "without",
    "b/c": "because",
}

# Conversational filler prefixes that can obscure syntactic intent
CONVERSATIONAL_FILLERS = [
    r"^(?:hey|hello|hi)\s+(?:doc|doctor|dr|team|there)[,\s]+",
    r"^(?:can\s+anyone\s+tell\s+me|could\s+someone\s+tell\s+me)[,\s]+",
    r"^(?:i\s+was\s+wondering|just\s+wondering)[,\s]+",
    r"^(?:please\s+help\s+me|help\s+me\s+understand)[,\s]+",
    r"^(?:so\s+basically|basically)[,\s]+"
]

COMMON_ENGLISH_WORDS = {
    "drinking", "walking", "though", "started", "waking", "several", "times", "water",
    "eating", "anything", "nothing", "having", "running", "feeling", "constantly",
    "bathroom", "toilet", "minutes", "always", "extremely", "nonstop", "through",
    "because", "without", "someone", "anyone", "wondering", "basically", "problem",
    "around", "before", "between", "during", "little", "really", "should", "people",
    "family", "person", "doctor", "health", "system", "change", "number", "follow",
    "another", "mother", "father", "friend", "taking", "making", "saying", "getting",
    "urinating", "urinate", "thirsty", "thirst", "cause", "causes", "caused", "causing",
    "headache", "headaches", "symptom", "symptoms", "treatment", "treatments",
    "prevent", "prevention", "diagnose", "diagnosis", "disease", "diseases"
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes basic Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class QueryNormalizer:
    def __init__(self):
        self.medical_vocabulary: Set[str] = set()
        self._init_vocabulary()

    def _init_vocabulary(self):
        # Base core medical conditions & anatomy vocabulary
        core_terms = [
            "diabetes", "glaucoma", "migraine", "asthma", "hypertension", "cholesterol",
            "pneumonia", "bronchitis", "arthritis", "thyroid", "leukemia", "osteoporosis",
            "arrhythmia", "depression", "epilepsy", "eczema", "psoriasis", "dermatitis",
            "hepatitis", "stroke", "aneurysm", "insomnia", "anxiety", "infection",
            "headache", "stomach", "diarrhea", "nausea", "fever", "urination", "polydipsia",
            "polyuria", "dyspnea", "palpitation", "allergy", "tuberculosis", "cardiovascular"
        ]
        self.medical_vocabulary.update(core_terms)

        # Attempt to load extra focus areas from medquad.csv if accessible
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        medquad_path = os.path.join(base_dir, "datasets", "medquad.csv")
        if os.path.exists(medquad_path):
            try:
                import pandas as pd
                df = pd.read_csv(medquad_path, usecols=["focus_area"])
                for fa in df["focus_area"].dropna().unique():
                    for token in re.findall(r"[a-zA-Z]{5,}", str(fa).lower()):
                        self.medical_vocabulary.add(token)
            except Exception:
                pass

    def normalize(self, query: str) -> Dict[str, Any]:
        """
        Normalizes a clinical or conversational query.
        Returns original_text, normalized_text, corrections list, and has_corrections boolean.
        """
        original = str(query).strip()
        working = original
        corrections: List[Dict[str, str]] = []

        # 1. Strip conversational filler prefix if present
        cleaned_prefix = working
        for pat in CONVERSATIONAL_FILLERS:
            match = re.search(pat, cleaned_prefix, flags=re.IGNORECASE)
            if match:
                matched_str = match.group(0)
                cleaned_prefix = re.sub(pat, "", cleaned_prefix, flags=re.IGNORECASE).strip()
                corrections.append({
                    "original": matched_str.strip(),
                    "replacement": "",
                    "type": "conversational_filler_removal"
                })
                break

        # 2. Tokenize words while preserving punctuation positions
        tokens = re.findall(r"\b[\w'/]+\b|[^\w\s]", cleaned_prefix)
        normalized_tokens = []

        for token in tokens:
            lower_token = token.lower()

            # Check abbreviation mapping
            if lower_token in COMMON_ABBREVIATIONS:
                replacement = COMMON_ABBREVIATIONS[lower_token]
                normalized_tokens.append(replacement)
                corrections.append({
                    "original": token,
                    "replacement": replacement,
                    "type": "abbreviation"
                })
                continue

            # Check high-confidence medical typo mapping
            if lower_token in COMMON_MEDICAL_TYPOS:
                replacement = COMMON_MEDICAL_TYPOS[lower_token]
                normalized_tokens.append(replacement)
                corrections.append({
                    "original": token,
                    "replacement": replacement,
                    "type": "spelling"
                })
                continue

            # Fuzzy match check for potential medical typos (len >= 6 and not standard English word)
            if len(lower_token) >= 6 and lower_token.isalpha() and lower_token not in COMMON_ENGLISH_WORDS:
                matched_med_word = None
                for med_word in self.medical_vocabulary:
                    # Filter: first letter must match and length difference <= 2
                    if med_word[0] == lower_token[0] and abs(len(med_word) - len(lower_token)) <= 2:
                        dist = levenshtein_distance(lower_token, med_word)
                        if dist == 1:
                            matched_med_word = med_word
                            break

                if matched_med_word and matched_med_word != lower_token:
                    normalized_tokens.append(matched_med_word)
                    corrections.append({
                        "original": token,
                        "replacement": matched_med_word,
                        "type": "fuzzy_spelling"
                    })
                    continue

            # Default: keep token unchanged
            normalized_tokens.append(token)

        # Reconstruct normalized string cleanly
        normalized_str = ""
        for i, t in enumerate(normalized_tokens):
            if i > 0 and re.match(r"\w", t) and re.match(r"\w", normalized_tokens[i-1]):
                normalized_str += " " + t
            elif re.match(r"[.,?!;:)]", t):
                normalized_str += t
            elif i > 0 and normalized_tokens[i-1] in "([":
                normalized_str += t
            else:
                if i > 0 and not normalized_str.endswith(" "):
                    normalized_str += " "
                normalized_str += t

        normalized_str = normalized_str.strip()
        # Capitalize first letter if needed
        if normalized_str:
            normalized_str = normalized_str[0].upper() + normalized_str[1:]

        return {
            "original_text": original,
            "normalized_text": normalized_str,
            "corrections": corrections,
            "has_corrections": len(corrections) > 0
        }


# Singleton instance
_NORMALIZER_INSTANCE = None

def get_normalizer() -> QueryNormalizer:
    global _NORMALIZER_INSTANCE
    if _NORMALIZER_INSTANCE is None:
        _NORMALIZER_INSTANCE = QueryNormalizer()
    return _NORMALIZER_INSTANCE
