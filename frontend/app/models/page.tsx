"use client";

import React, { useState, useEffect } from "react";
import {
  Brain,
  Layers,
  Activity,
  Cpu,
  Database,
  CheckCircle2,
  GitBranch,
  ArrowDown,
  Sparkles,
  BarChart,
  Code
} from "lucide-react";
import { fetchModelInfo } from "@/lib/api";
import { ModelInfo } from "@/types";

export default function ModelsPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      
      {/* Title */}
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-semibold mb-3">
          <Layers className="w-3.5 h-3.5" />
          Technical Specifications & Deep Learning Architecture
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Model Engineering & Benchmarks
        </h1>
        <p className="mt-2 text-sm sm:text-base text-slate-400">
          A comparative analysis of Traditional NLP, Bidirectional Recurrent Networks, and Multi-Task Contextual Transformers for clinical question classification.
        </p>
      </div>

      {/* Architecture Flow Diagram */}
      <div className="glass-panel p-8 sm:p-10 rounded-3xl border border-slate-800 mb-12">
        <h3 className="text-base font-bold text-white mb-6 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-brand-400" />
          End-to-End System Pipeline Flow
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center text-xs">
          
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col items-center">
            <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-slate-300 font-bold mb-3">
              1
            </div>
            <h4 className="font-bold text-white text-sm mb-1">Input Ingestion</h4>
            <p className="text-slate-400">
              Text normalization, punctuation standardization, regex cleaning, and emergency keyword triage.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col items-center">
            <div className="w-10 h-10 rounded-xl bg-indigo-950 border border-indigo-500/30 flex items-center justify-center text-indigo-300 font-bold mb-3">
              2
            </div>
            <h4 className="font-bold text-white text-sm mb-1">Domain Gate</h4>
            <p className="text-slate-400">
              Contextual Domain Classifier (Medical vs Non-Medical OOD). 100% test accuracy on filtering non-clinical inquiries.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-brand-950/40 border border-brand-500/40 flex flex-col items-center shadow-lg shadow-brand-500/10">
            <div className="w-10 h-10 rounded-xl bg-brand-500/20 border border-brand-500/40 flex items-center justify-center text-brand-300 font-bold mb-3">
              3
            </div>
            <h4 className="font-bold text-brand-300 text-sm mb-1">Transformer Encoder</h4>
            <p className="text-slate-300">
              BERT contextual encoder + Mean & CLS pooling + LayerNorm projection into 256-dimensional latent representation.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col items-center">
            <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-slate-300 font-bold mb-3">
              4
            </div>
            <h4 className="font-bold text-white text-sm mb-1">Multi-Task Heads</h4>
            <p className="text-slate-400">
              Joint softmax heads predict Specialty Category (13 classes), Intention (10 classes), Uncertainty, and retrieve top similar questions.
            </p>
          </div>

        </div>

        {/* Multi-Task Mathematical Loss Formulation */}
        {/* Multi-Task Mathematical Loss Formulation */}
        <div className="mt-8 pt-6 border-t border-slate-800/80 bg-slate-950/50 p-6 rounded-2xl border border-slate-800">
          <div className="flex items-center gap-2 text-xs font-mono text-brand-400 mb-2">
            <Code className="w-4 h-4" />
            MULTI-TASK OPTIMIZATION OBJECTIVE
          </div>
          <div className="font-mono text-sm sm:text-base text-white p-3 rounded-xl bg-slate-900/80 border border-slate-800 overflow-x-auto">
            {"\\mathcal{L}_{total} = \\mathcal{L}_{domain(weighted)} + 1.2 \\times \\mathcal{L}_{category} + 1.0 \\times \\mathcal{L}_{intent}"}
          </div>
          <p className="mt-3 text-xs text-slate-400 leading-relaxed">
            Where each component is cross-entropy loss over its respective ground truth label. Domain loss uses 8.0× class-weighted penalty to balance medical vs non-medical classification, while Category loss is weighted at 1.2 to balance gradient flow across clinical specialties and fine-grained communication intents.
          </p>
        </div>
      </div>

      {/* Model Evolution Benchmark Comparison Table */}
      <div className="glass-panel p-8 sm:p-10 rounded-3xl border border-slate-800 mb-12">
        <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
          <Activity className="w-5 h-5 text-brand-400" />
          Empirical Benchmark Results (Test Set Evaluation)
        </h3>
        <p className="text-xs text-slate-400 mb-6">
          Rigorous evaluation across traditional ML, recurrent deep learning, and contextual transformers evaluated on 957 held-out test questions from the expanded 9,566-sample dataset.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-4">Domain Acc</th>
                <th className="py-3 px-4">Category Acc</th>
                <th className="py-3 px-4">Category Macro F1</th>
                <th className="py-3 px-4">Intent Acc</th>
                <th className="py-3 px-4">Intent Macro F1</th>
                <th className="py-3 px-4">Inference Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              
              <tr className="hover:bg-slate-900/40">
                <td className="py-3.5 px-4 font-semibold text-slate-300">
                  Baseline: TF-IDF + Linear SVM
                </td>
                <td className="py-3.5 px-4 font-mono text-slate-200">100.0%</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">97.97%</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">0.9525</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">97.29%</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">0.8686</td>
                <td className="py-3.5 px-4 font-mono text-slate-400">~1.2 ms</td>
              </tr>

              <tr className="hover:bg-slate-900/40">
                <td className="py-3.5 px-4 font-semibold text-slate-300">
                  Deep Learning: 2-Layer BiLSTM
                </td>
                <td className="py-3.5 px-4 font-mono text-slate-200">100.0%</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">94.46%</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">0.9258</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">99.77%</td>
                <td className="py-3.5 px-4 font-mono text-slate-200">0.9980</td>
                <td className="py-3.5 px-4 font-mono text-slate-400">~4.5 ms</td>
              </tr>

              <tr className="bg-brand-950/30 border-l-2 border-brand-400">
                <td className="py-3.5 px-4 font-bold text-brand-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-brand-400" />
                  Transformer SOTA: Multi-Task BERT
                </td>
                <td className="py-3.5 px-4 font-mono text-brand-300 font-bold">100.0%</td>
                <td className="py-3.5 px-4 font-mono text-brand-300 font-bold">98.42%</td>
                <td className="py-3.5 px-4 font-mono text-brand-300 font-bold">0.9643</td>
                <td className="py-3.5 px-4 font-mono text-brand-300 font-bold">99.66%</td>
                <td className="py-3.5 px-4 font-mono text-brand-300 font-bold">0.8833</td>
                <td className="py-3.5 px-4 font-mono text-slate-300">~10.8 ms</td>
              </tr>

            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset & Taxonomy Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-800">
          <h3 className="text-base font-bold text-white mb-3 flex items-center gap-2">
            <Database className="w-5 h-5 text-brand-400" />
            Curated NIH MedQuAD Dataset
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed mb-4">
            The dataset was curated from NIH sources (NINDS, NHLBI, NIDDK, CancerGov, GHR, GARD, CDC). Cleaned of duplicates and balanced across 13 clinical specialties:
          </p>
          <div className="space-y-2 text-xs text-slate-400">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Total Curated Dataset Size:</span>
              <strong className="text-white">9,566 Questions</strong>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Training Split (80% Stratified):</span>
              <span className="text-slate-200">7,652 samples</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Validation Split (10%):</span>
              <span className="text-slate-200">957 samples</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Test Split (10% Held-Out):</span>
              <span className="text-slate-200">957 samples</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Indexed NIH Q&A Knowledge Base:</span>
              <span className="text-brand-300 font-semibold">8,000 cases</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Data Leakage Verification:</span>
              <span className="text-emerald-400 font-semibold">0 duplicate overlap</span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-800">
          <h3 className="text-base font-bold text-white mb-3 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-accent-400" />
            Taxonomy Classes Supported
          </h3>
          <div className="space-y-4">
            <div>
              <span className="text-xs font-semibold uppercase text-brand-400 block mb-2">
                13 Medical Specialty Categories:
              </span>
              <div className="flex flex-wrap gap-1.5 text-[11px]">
                {[
                  "Cardiology", "Neurology", "Respiratory", "Gastroenterology",
                  "Endocrinology", "Oncology", "Dermatology", "Infectious Disease",
                  "Orthopedics & Rheumatology", "Mental Health", "Medication & Pharmacology",
                  "Genetics & Rare Diseases", "General Medicine"
                ].map((c, i) => (
                  <span key={i} className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    {c}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <span className="text-xs font-semibold uppercase text-accent-400 block mb-2">
                10 Clinical Intentions:
              </span>
              <div className="flex flex-wrap gap-1.5 text-[11px]">
                {[
                  "Symptoms", "Causes", "Diagnosis", "Treatment", "Prevention",
                  "Risk Factors", "Prognosis", "Side Effects", "Medication", "General Information"
                ].map((it, i) => (
                  <span key={i} className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    {it}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
