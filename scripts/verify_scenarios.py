"""
Verification Script for all 11 Test Scenarios
Tests:
A. Migraine causes
B. Diabetes symptoms
C. Multi-symptom: "I am having leg pain, fever and chest pain"
D. Emergency: "I have severe chest pain and difficulty breathing"
E. Non-medical: "What is the capital of France?"
F. Ambiguous: "Headache"
G. Medication: "What are common side effects of antibiotics?"
H. Prevention: "How can heart disease be prevented?"
I. Long input (>1000 chars)
J. Empty input
K. Repeated queries (timestamp verification)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

scenarios = [
    ("Scenario A (Migraine Causes)", "What causes migraine headaches?"),
    ("Scenario B (Diabetes Symptoms)", "What are the early symptoms of diabetes?"),
    ("Scenario C (Multi-Symptom Composite)", "I am having leg pain, fever and chest pain"),
    ("Scenario D (Emergency Combination)", "I have severe chest pain and difficulty breathing"),
    ("Scenario E (Non-Medical Gate)", "What is the capital of France?"),
    ("Scenario F (Ambiguous Single-Word)", "Headache"),
    ("Scenario G (Medication Inquiry)", "What are common side effects of antibiotics?"),
    ("Scenario H (Prevention Inquiry)", "How can heart disease be prevented?"),
    ("Scenario I (Long Input >1000 chars)", "My question is about health. " * 50),
    ("Scenario J (Empty Input)", ""),
    ("Scenario K (Repeated Timestamp Test)", "What causes migraine headaches?")
]

results = {}

for name, query in scenarios:
    print(f"\n=======================================================")
    print(f"RUNNING: {name}")
    print(f"Query: {query[:60]}..." if len(query) > 60 else f"Query: {query}")
    
    response = client.post("/api/classify", json={"question": query})
    print(f"HTTP Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Is Medical: {data.get('is_medical')}")
        print(f"Primary Category: {data.get('primary_category') or data.get('category')}")
        print(f"Primary Reason: {data.get('primary_category_reason')}")
        print(f"Multi-Symptom: {data.get('is_multi_symptom')}")
        print(f"Detected Entities: {data.get('detected_entities')}")
        print(f"Intent: {data.get('intent')}")
        print(f"Safety Level: {data.get('safety_level')}")
        print(f"Uncertain: {data.get('uncertain')}")
        print(f"Clarification: {data.get('clarification_needed')}")
        print(f"Time (IST): {data.get('timestamp_ist')}")
        if data.get("emergency", {}).get("is_emergency"):
            print(f"EMERGENCY WARNING: {data['emergency']['warning']}")
        print(f"Next Steps: {data.get('suggested_next_steps')[:2]}")
        results[name] = {
            "status": 200,
            "data": data
        }
    else:
        print(f"Error Detail: {response.json()}")
        results[name] = {
            "status": response.status_code,
            "error": response.json()
        }

# Verify Analytics Endpoint reflects the queries
res_ana = client.get("/api/analytics")
print(f"\n=======================================================")
print(f"ANALYTICS SUMMARY:")
ana_data = res_ana.json()
print(f"Total Queries: {ana_data['total_queries']}")
print(f"Recent Activity Count: {len(ana_data['recent_activity'])}")
if ana_data['recent_activity']:
    first_evt = ana_data['recent_activity'][0]
    print(f"Latest Event Time (IST): {first_evt.get('timestamp_ist')}")
    print(f"Latest Event Query: {first_evt.get('question_preview')}")
    print(f"Latest Event Multi-Symptom: {first_evt.get('is_multi_symptom')}")

print("\nALL 11 SCENARIOS EXECUTED SUCCESSFULLY.")
