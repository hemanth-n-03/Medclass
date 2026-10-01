import React from "react";
import Link from "next/link";
import { Brain, ShieldAlert, Heart, Terminal, Database, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950/90 text-slate-400 py-12 px-4 sm:px-6 lg:px-8 mt-auto">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Safety Disclaimer Banner */}
        <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 flex flex-col sm:flex-row items-start sm:items-center gap-3 text-amber-200/90 text-xs leading-relaxed">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
          <p>
            <strong className="font-semibold text-amber-300">Responsible AI & Clinical Disclaimer:</strong> MedClass AI is an NLP classification prototype developed for medical question intelligence and taxonomy mapping. It does <span className="underline decoration-amber-400 font-semibold">NOT</span> diagnose conditions, prescribe medications, or replace certified clinical judgment. If you are experiencing a medical emergency, immediately contact emergency services: 112 (National Emergency Helpline) or 108 (Ambulance).
          </p>
        </div>

        {/* Footer Content Columns */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-brand-400" />
              <span className="font-bold text-base text-white">MedClass AI</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Autonomous Deep Learning & Multi-Task Transformer for clinical question classification, intention extraction, and semantic retrieval.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-3">Model Core</h4>
            <ul className="space-y-2 text-xs">
              <li className="flex items-center gap-2 text-slate-400">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                BERT Contextual Multi-Task Encoder
              </li>
              <li className="flex items-center gap-2 text-slate-400">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                2-Layer BiLSTM Deep Baseline
              </li>
              <li className="flex items-center gap-2 text-slate-400">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                Calibrated Linear SVM ML Baseline
              </li>
              <li className="flex items-center gap-2 text-slate-400">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                NIH MedQuAD Curated Corpus
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-3">Platform</h4>
            <ul className="space-y-2 text-xs">
              <li>
                <Link href="/classify" className="hover:text-brand-300 transition-colors">Question Classifier</Link>
              </li>
              <li>
                <Link href="/analytics" className="hover:text-brand-300 transition-colors">Analytics & Query Audit</Link>
              </li>
              <li>
                <Link href="/models" className="hover:text-brand-300 transition-colors">Benchmark Metrics</Link>
              </li>
              <li>
                <Link href="/about" className="hover:text-brand-300 transition-colors">Ethics & Safety Policy</Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-3">Architecture & Standards</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              End-to-end clinical NLP intelligence system adhering to Responsible AI safety guidelines, multi-symptom signal routing, and verified medical knowledge retrieval.
            </p>
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
              <span>FastAPI Backend</span>
              <span>•</span>
              <span>Next.js Frontend</span>
            </div>
          </div>
        </div>

        {/* Bottom copyright */}
        <div className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-4">
          <p>© 2026 MedClass AI. Built for Medical Question Intelligence & Clinical NLP.</p>
          <div className="flex items-center gap-4">
            <span className="inline-flex items-center gap-1">
              Made with <Heart className="w-3.5 h-3.5 text-rose-500 fill-rose-500" /> for Healthcare Innovation
            </span>
          </div>
        </div>

      </div>
    </footer>
  );
}
