"""
Baseline Machine Learning Model Training & Evaluation
Uses TF-IDF with Logistic Regression and Linear SVM for:
1. Medical vs Non-Medical Binary Classification
2. Medical Category Classification
3. Medical Intent Classification
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models", "baseline")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_eval_task(X_train, y_train, X_val, y_val, X_test, y_test, task_name, classes=None):
    print(f"\n==========================================")
    print(f"Training Baseline for Task: {task_name.upper()}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # 1. TF-IDF Representation
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # 2. Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)

    # 3. Linear SVM (Calibrated for probability estimation)
    svm_base = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svm_cal = CalibratedClassifierCV(estimator=svm_base)
    svm_cal.fit(X_train_vec, y_train)
    y_pred_svm = svm_cal.predict(X_test_vec)

    # Calculate metrics
    def compute_metrics(y_true, y_pred):
        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_f1": round(f1_weighted, 4)
        }

    lr_metrics = compute_metrics(y_test, y_pred_lr)
    svm_metrics = compute_metrics(y_test, y_pred_svm)

    print(f"Logistic Regression Results for {task_name}:", lr_metrics)
    print(f"Linear SVM Results for {task_name}:", svm_metrics)

    # Choose best model (by macro_f1)
    best_name = "Logistic Regression" if lr_metrics["macro_f1"] >= svm_metrics["macro_f1"] else "Linear SVM"
    best_model = lr if best_name == "Logistic Regression" else svm_cal
    best_preds = y_pred_lr if best_name == "Logistic Regression" else y_pred_svm
    best_metrics = lr_metrics if best_name == "Logistic Regression" else svm_metrics

    # Confusion matrix & full report
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, best_preds, labels=labels).tolist()
    report = classification_report(y_test, best_preds, labels=labels, output_dict=True, zero_division=0)

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, f"{task_name}_vectorizer.joblib"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{task_name}_model.joblib"))

    return {
        "best_algorithm": best_name,
        "metrics": best_metrics,
        "lr_metrics": lr_metrics,
        "svm_metrics": svm_metrics,
        "labels": labels,
        "confusion_matrix": cm,
        "classification_report": report
    }

def run_baseline_training():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    results = {}

    # Task 1: Medical vs Non-Medical
    med_res = train_and_eval_task(
        train_df["clean_question"], train_df["is_medical"].astype(int),
        val_df["clean_question"], val_df["is_medical"].astype(int),
        test_df["clean_question"], test_df["is_medical"].astype(int),
        task_name="domain"
    )
    results["domain"] = med_res

    # Task 2: Category Classification (on Medical subset)
    train_med = train_df[train_df["is_medical"] == 1]
    val_med = val_df[val_df["is_medical"] == 1]
    test_med = test_df[test_df["is_medical"] == 1]

    cat_res = train_and_eval_task(
        train_med["clean_question"], train_med["category"],
        val_med["clean_question"], val_med["category"],
        test_med["clean_question"], test_med["category"],
        task_name="category"
    )
    results["category"] = cat_res

    # Task 3: Intent Classification (on Medical subset)
    intent_res = train_and_eval_task(
        train_med["clean_question"], train_med["intent"],
        val_med["clean_question"], val_med["intent"],
        test_med["clean_question"], test_med["intent"],
        task_name="intent"
    )
    results["intent"] = intent_res

    # Save results to json
    metrics_path = os.path.join(MODEL_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll baseline results saved to {metrics_path}")

    return results

if __name__ == "__main__":
    run_baseline_training()
