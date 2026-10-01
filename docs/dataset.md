# MedClass AI — Dataset Engineering & Quality Pipeline

## 1. Data Origin & Provenance
The clinical inquiry corpus for MedClass AI is derived from **MedQuAD (Medical Question Answering Dataset)**, compiled by the National Institutes of Health (NIH) and the National Library of Medicine (NLM).

### Sources Included:
- **NINDS:** National Institute of Neurological Disorders and Stroke
- **NHLBI:** National Heart, Lung, and Blood Institute
- **NIDDK:** National Institute of Diabetes and Digestive and Kidney Diseases
- **CancerGov:** National Cancer Institute
- **GHR / GARD:** Genetics Home Reference & Rare Diseases
- **MPlusHealthTopics:** MedlinePlus Consumer Health Topics
- **CDC:** Centers for Disease Control and Prevention

---

## 2. Preprocessing & Data Cleaning
Raw MedQuAD files contain template artifacts and duplicates. The automated pipeline (`ml/preprocessing/build_dataset.py`) enforces:
1. **HTML & Tag Stripping:** Removal of tags and legacy markup.
2. **Punctuation Normalization:** Correction of spaced punctuation (e.g. `Glaucoma ? ?` → `Glaucoma?`).
3. **Template Normalization:** Replacement of formulaic templates (e.g. `What is (are) X ?` → `What is X?`).
4. **Deduplication:** Removal of 1,427 exact duplicate question templates.
5. **Class Balancing:** Balanced sub-sampling ensuring rare categories maintain viable sample density without being eclipsed by genetics syndromes.

---

## 3. Taxonomy Design

### Medical Categories (13 Specialties):
1. **Cardiology** (Arrhythmia, hypertension, heart attacks, coronary disease)
2. **Neurology** (Stroke, epilepsy, dementia, Parkinson's, migraines)
3. **Respiratory** (Asthma, COPD, bronchitis, pulmonary disease)
4. **Gastroenterology** (Cirrhosis, Crohn's, colitis, GERD, digestive health)
5. **Endocrinology** (Diabetes, thyroid disorders, hormones, Cushing's)
6. **Oncology** (Leukemia, tumors, lymphomas, chemotherapy)
7. **Dermatology** (Psoriasis, eczema, rashes, skin disorders)
8. **Infectious Disease** (Sepsis, viral infections, HIV, tuberculosis, flu)
9. **Orthopedics & Rheumatology** (Arthritis, osteoporosis, fractures, lupus)
10. **Mental Health** (Depression, anxiety, psychiatric conditions)
11. **Medication & Pharmacology** (Prescription drugs, antibiotic usage, dosages)
12. **Genetics & Rare Diseases** (Chromosomal abnormalities, inherited syndromes)
13. **General Medicine** (Preventive health, fatigue, multi-system wellness)

### Clinical Intentions (10 Classes):
1. **Symptoms** (Signs, early warnings, manifestations)
2. **Causes** (Etiology, triggers, genetic mutations)
3. **Diagnosis** (Biopsy, screening, laboratory tests, exams)
4. **Treatment** (Therapies, surgeries, management protocols)
5. **Prevention** (Diet, lifestyle, vaccination, avoidance)
6. **Risk Factors** (Predispositions, demographics at risk)
7. **Prognosis** (Outlook, disease stages, complications)
8. **Side Effects** (Adverse drug interactions, contraindications)
9. **Medication** (Drug therapies, dosages, pharmaceutical indications)
10. **General Information** (Disease overview, prevalence, anatomy)

---

## 4. Contrastive Out-Of-Domain Corpus
To train the Stage-1 Domain Gate, an open-domain contrastive set of 410 non-medical questions was integrated spanning:
- Computer engineering & IT troubleshooting ("How do I repair my laptop battery?")
- Physics & space science ("Why is the sky blue?")
- Geography & world history ("What is the capital of India?")
- Cooking & daily life ("How do I bake chocolate chip cookies?")
- Finance & economics ("How do I file taxes online?")

---

## 5. Dataset Split & Leakage Verification
- **Total Records:** 5,125
- **Train Split (80% Stratified):** 4,100 samples
- **Validation Split (10%):** 512 samples
- **Test Split (10% Held-Out):** 513 samples
- **Data Leakage Check:** $\text{Overlap}(\text{Train}, \text{Test}) = 0$. Zero sample contamination verified.
