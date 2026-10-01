"""
Data Preprocessing & Dataset Construction Pipeline for MedClass AI
Builds a curated, deduplicated, clinically labeled multi-task medical question dataset
from MedQuAD plus a diverse open-domain non-medical dataset.
"""

import json
import os
import re
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seed for full reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
RAW_MEDQUAD_PATH = os.path.join(DATA_DIR, "medquad.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
STATS_PATH = os.path.join(DATA_DIR, "dataset_stats.json")

# Clean question text
def clean_question_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove HTML entities or formatting artifacts
    text = re.sub(r"<[^>]+>", "", text)
    # Fix spacing around punctuation: "Glaucoma ?" -> "Glaucoma?"
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    # Fix double punctuation: "??", "? ?"
    text = re.sub(r"\?+", "?", text)
    # Clean redundant parentheses in MedQuAD templates: "What is (are)" -> "What is"
    text = re.sub(r"What is \(are\)", "What is", text, flags=re.IGNORECASE)
    text = re.sub(r"How to (prevent|treat)", r"How can you \1", text, flags=re.IGNORECASE)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Detect clinical intent
def extract_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["symptom", "signs", "warning sign", "early sign", "manifestation"]):
        return "Symptoms"
    if any(k in ql for k in ["cause", "why do", "genetic change", "inherited", "etiology", "pathogen", "origin"]):
        return "Causes"
    if any(k in ql for k in ["treat", "cure", "therapy", "what to do for", "manage", "intervention", "surgery", "remedy"]):
        return "Treatment"
    if any(k in ql for k in ["diagnos", "test", "exam", "screen", "biopsy", "mri", "scan", "detect"]):
        return "Diagnosis"
    if any(k in ql for k in ["prevent", "avoid", "protect against", "reduce risk"]):
        return "Prevention"
    if any(k in ql for k in ["risk factor", "who is at risk", "susceptib", "more likely", "predispos"]):
        return "Risk Factors"
    if any(k in ql for k in ["complication", "outlook", "prognos", "stage", "long-term", "survival", "expect"]):
        return "Prognosis"
    if any(k in ql for k in ["side effect", "adverse", "reaction", "contraindicat"]):
        return "Side Effects"
    if any(k in ql for k in ["medication", "drug", "dose", "dosage", "pill", "prescript", "antibiotic", "medicine"]):
        return "Medication"
    return "General Information"

# Map medical specialty category
def map_category(row) -> str:
    focus = str(row.get("focus_area", "")).lower()
    src = str(row.get("source", "")).lower()
    q = str(row.get("question", "")).lower()
    text = f"{focus} {src} {q}"

    # Category matching with prioritized clinical keywords
    if any(k in text for k in ["heart", "cardio", "coronary", "hypertension", "blood pressure", "arrhythmia", "cholesterol", "vascular", "aortic", "atherosclerosis", "cardiac"]):
        return "Cardiology"
    if any(k in text for k in ["brain", "stroke", "neuro", "alzheimer", "parkinson", "epilepsy", "migraine", "headache", "dementia", "sclerosis", "neuropathy", "cerebral", "tremor"]):
        return "Neurology"
    if any(k in text for k in ["lung", "respiratory", "asthma", "copd", "pneumonia", "bronch", "pulmonary", "breath", "cystic fibrosis", "emphysema"]):
        return "Respiratory"
    if any(k in text for k in ["gastro", "stomach", "liver", "bowel", "colon", "cirrhosis", "crohn", "colitis", "gerd", "hepatitis", "digest", "pancrea", "esophag", "ulcer"]):
        return "Gastroenterology"
    if any(k in text for k in ["diabet", "thyroid", "endocrine", "adrenal", "hormon", "metabolic", "pituitary", "insulin", "cushing", "addison", "hypothyroid"]):
        return "Endocrinology"
    if any(k in text for k in ["cancer", "tumor", "leukemia", "lymphoma", "carcinoma", "melanoma", "oncolog", "neoplasm", "sarcoma", "myeloma", "malignan"]):
        return "Oncology"
    if any(k in text for k in ["skin", "dermat", "rash", "eczema", "psoriasis", "shingles", "acne", "vitiligo", "alopecia", "dermis", "hives"]):
        return "Dermatology"
    if any(k in text for k in ["infect", "virus", "viral", "bacteria", "sepsis", "hiv", "aids", "influenza", "covid", "tuberculosis", "parasit", "fungal", "strep"]):
        return "Infectious Disease"
    if any(k in text for k in ["bone", "joint", "arthrit", "osteo", "orthoped", "spine", "fracture", "lupus", "rheumat", "fibromyalgia", "musculoskeletal", "tendon"]):
        return "Orthopedics & Rheumatology"
    if any(k in text for k in ["mental", "depress", "anxiety", "psych", "bipolar", "schizophrenia", "autism", "adhd", "stress", "addiction", "panic"]):
        return "Mental Health"
    if any(k in text for k in ["drug", "medicat", "antibiotic", "dosage", "pharmac", "pill", "prescript", "vaccin", "aspirin", "ibuprofen"]):
        return "Medication & Pharmacology"
    if any(k in text for k in ["gene", "genetic", "heredit", "chromosom", "inherited", "mutation", "syndrome", "dystrophy", "anomal"]):
        return "Genetics & Rare Diseases"
    return "General Medicine"

# Rich set of non-medical benchmark questions for OOD and medical/non-medical binary stage
def generate_non_medical_questions() -> list:
    non_med_seeds = [
        # Tech & Computing
        "How do I repair my laptop battery?",
        "How do I reset my Wi-Fi router password?",
        "What is the difference between Python and Java?",
        "How do I install Node.js on Windows 11?",
        "Why is my laptop screen flickering randomly?",
        "How do I write a binary search algorithm in C++?",
        "What is Git rebase and how is it used?",
        "How do I configure SSL certificate in Nginx?",
        "Why is my graphics card fan running so loud?",
        "How do I recover deleted files from a USB flash drive?",
        "What is Docker containerization and how does it work?",
        "How do I set up a PostgreSQL database connection?",
        "Why is my phone charging so slowly?",
        "How do I clear browser cache in Google Chrome?",
        "What is the best way to learn machine learning mathematics?",
        "How do I fix a blue screen of death on Windows?",
        "How do I build a REST API in FastAPI?",
        "What is cloud computing with AWS?",
        "How do I format an external hard drive on MacOS?",
        "Why does my computer freeze when playing video games?",
        # General Science & Physics & Space
        "What is the theory of general relativity?",
        "Why is the sky blue during midday?",
        "How does photosynthesis convert sunlight into energy?",
        "What is the distance between the Earth and the Moon?",
        "How do black holes form in deep space?",
        "What is quantum entanglement and how does it occur?",
        "Why do tectonic plates shift and cause earthquakes?",
        "How do solar panels convert light into electricity?",
        "What is the speed of light in a vacuum?",
        "How do airplanes generate lift during takeoff?",
        "What causes the northern lights and aurora borealis?",
        "How old is the universe according to cosmic measurements?",
        # Geography, History & Politics
        "What is the capital of France?",
        "What is the capital of India?",
        "What is the capital of Germany?",
        "What is the capital of Japan?",
        "What is the capital of Canada?",
        "What is the capital of Australia?",
        "What is the currency of the United Kingdom?",
        "Who painted the Mona Lisa?",
        "What is the tallest mountain in the world?",
        "When did the Second World War officially end?",
        "What is the longest river in the world?",
        "Who was the first president of the United States?",
        "Where are the Great Pyramids of Giza located?",
        "What is the population of Tokyo in 2026?",
        "How many continents are there on Earth?",
        "What caused the fall of the Roman Empire?",
        "Who discovered the route to the Americas in 1492?",
        "What is the deepest point in the world's oceans?",
        # Entertainment, Sports & Pop Culture
        "Who won yesterday's cricket match?",
        "Who won the FIFA World Cup in 2022?",
        "How do I learn to play the acoustic guitar for beginners?",
        "Who directed the movie Inception?",
        "How many players are on a basketball team on the court?",
        "What is the plot of the novel 1984 by George Orwell?",
        "Who wrote the Harry Potter book series?",
        "What are the rules of Olympic curling?",
        "How do I compose music in digital audio workstations?",
        "What is the history of the modern Olympic Games?",
        # Cooking, Home & Daily Life
        "How do I bake chocolate chip cookies from scratch?",
        "How do I change a flat tire on a highway?",
        "What is the easiest way to remove coffee stains from clothes?",
        "How do I brew cold brew coffee at home?",
        "What temperature should chicken be cooked to?",
        "How do I unclog a bathroom sink without harsh chemicals?",
        "How do I train a puppy to stop barking?",
        "What are the best indoor plants that require low sunlight?",
        "How do I clean an oven with baking soda and vinegar?",
        "How do I iron a formal dress shirt properly?",
        # Business, Finance & Law
        "How do I file my personal income taxes online?",
        "What is compound interest and how does it accumulate?",
        "What is the difference between stocks and corporate bonds?",
        "How does inflation affect the average purchasing power?",
        "What is a mutual fund and how is risk diversified?",
        "How do I register a limited liability company?",
        "What is mortgage refinancing and when does it make sense?",
        "How do credit scores work and how to improve them?",
        "What is the role of the central bank in monetary policy?",
        "What is cryptocurrency blockchain verification?",
        # Travel, Languages & Art
        "What are the top tourist attractions to visit in Paris?",
        "How can I quickly learn conversational Spanish?",
        "What is the difference between oil paint and acrylic paint?",
        "How do I apply for an international travel visa?",
        "What are the best places to hike in the Swiss Alps?",
        "How do I book cheap flights during holiday seasons?",
        "What is the history of the Renaissance art movement?",
        "How do you pronounce words in Mandarin Chinese?",
        "What is the architecture style of the Taj Mahal?",
        "How do I pack light for a two-week international trip?",
    ]

    # Generate realistic syntactical variations
    prefixes = [
        "Can you explain ",
        "I need help understanding ",
        "What can you tell me about ",
        "Could you guide me on ",
        "Where can I find information on ",
        "Tell me about ",
        "What is the process to ",
        "Is there a good way to ",
    ]
    expanded = []
    for q in non_med_seeds:
        expanded.append(q)
        clean_stem = re.sub(r"^(how do i |what is |why is |who is |where is )", "", q.lower()).rstrip("?")
        for p in prefixes[:7]:
            expanded.append(f"{p}{clean_stem}?")
    
    # Return unique cleaned list
    return list(dict.fromkeys(expanded))

def build_dataset():
    print(f"Loading raw MedQuAD from {RAW_MEDQUAD_PATH}...")
    df_raw = pd.read_csv(RAW_MEDQUAD_PATH)
    print(f"Raw rows: {len(df_raw)}")

    # 1. Cleaning & Normalization
    df_raw["clean_question"] = df_raw["question"].apply(clean_question_text)
    
    # 2. Filter invalid or malformed questions
    df_raw = df_raw[df_raw["clean_question"].str.len() >= 10].copy()
    
    # 3. Deduplication on cleaned text
    duplicates_before = df_raw.duplicated(subset=["clean_question"]).sum()
    print(f"Duplicates detected on clean question: {duplicates_before}")
    df_clean = df_raw.drop_duplicates(subset=["clean_question"]).copy()
    print(f"Unique medical questions after deduplication: {len(df_clean)}")

    # 4. Label derivation
    df_clean["intent"] = df_clean["clean_question"].apply(extract_intent)
    df_clean["category"] = df_clean.apply(map_category, axis=1)
    df_clean["is_medical"] = 1
    
    # Balance and scale dataset: include up to 2,500 unique samples per category
    # Every question remains completely unique (no duplication before split)
    sampled_medical = []
    for cat, group in df_clean.groupby("category"):
        n_samples = min(len(group), 2500)
        sampled_medical.append(group.sample(n=n_samples, random_state=SEED))
    
    df_med_balanced = pd.concat(sampled_medical, ignore_index=True)
    print(f"Expanded medical dataset: {len(df_med_balanced)} records across {df_med_balanced['category'].nunique()} categories")

    # 5. Non-medical questions
    non_med_qs = generate_non_medical_questions()
    df_non_med = pd.DataFrame({
        "question": non_med_qs,
        "clean_question": [clean_question_text(q) for q in non_med_qs],
        "answer": ["N/A"] * len(non_med_qs),
        "source": ["OpenDomain"] * len(non_med_qs),
        "focus_area": ["Non-Medical"] * len(non_med_qs),
        "intent": ["Non-Medical"] * len(non_med_qs),
        "category": ["Non-Medical"] * len(non_med_qs),
        "is_medical": [0] * len(non_med_qs),
    })
    print(f"Non-medical questions: {len(df_non_med)}")

    # Combine into full dataset
    df_full = pd.concat([df_med_balanced, df_non_med], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Total dataset size: {len(df_full)}")

    # 6. Stratified train/val/test split (80 / 10 / 10)
    # Stratify by category to maintain balance across splits
    train_df, temp_df = train_test_split(
        df_full,
        test_size=0.20,
        random_state=SEED,
        stratify=df_full["category"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["category"]
    )

    print(f"Splits created: Train={len(train_df)} (80%), Val={len(val_df)} (10%), Test={len(test_df)} (10%)")

    # 7. Check for data leakage
    train_qs = set(train_df["clean_question"])
    val_leakage = val_df["clean_question"].isin(train_qs).sum()
    test_leakage = test_df["clean_question"].isin(train_qs).sum()
    print(f"Data leakage check: Val in Train={val_leakage}, Test in Train={test_leakage} (Must be 0)")
    assert val_leakage == 0 and test_leakage == 0, "Data leakage detected between splits!"

    # 8. Save CSV files
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")

    # 9. Compute comprehensive statistics
    categories = sorted([c for c in df_full["category"].unique() if c != "Non-Medical"])
    intents = sorted([i for i in df_full["intent"].unique() if i != "Non-Medical"])

    stats = {
        "total_records": len(df_full),
        "medical_records": int((df_full["is_medical"] == 1).sum()),
        "non_medical_records": int((df_full["is_medical"] == 0).sum()),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "num_categories": len(categories),
        "num_intents": len(intents),
        "categories": categories,
        "intents": intents,
        "category_distribution": df_full["category"].value_counts().to_dict(),
        "intent_distribution": df_full["intent"].value_counts().to_dict(),
        "train_category_distribution": train_df["category"].value_counts().to_dict(),
        "test_category_distribution": test_df["category"].value_counts().to_dict(),
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics metadata to {STATS_PATH}")
    return stats

if __name__ == "__main__":
    build_dataset()
