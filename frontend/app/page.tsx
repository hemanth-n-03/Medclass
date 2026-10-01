"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Brain,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Search,
  Activity,
  Layers,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Stethoscope,
  HeartPulse,
  Database
} from "lucide-react";

export default function HomePage() {
  const router = useRouter();
  const [quickInput, setQuickInput] = useState("");

  const sampleQuestions = [
    "What are the early symptoms of diabetes?",
    "What causes migraine headaches?",
    "How is asthma diagnosed in children?",
    "What are common side effects of antibiotics?",
    "How can heart disease be prevented?"
  ];

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (quickInput.trim()) {
      router.push(`/classify?q=${encodeURIComponent(quickInput.trim())}`);
    }
  };

  const handleSelectSample = (sample: string) => {
    router.push(`/classify?q=${encodeURIComponent(sample)}`);
  };

  return (
    <div className="relative overflow-hidden flex flex-col items-center">
      
      {/* Background radial gradient glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] bg-brand-500/10 blur-[130px] rounded-full pointer-events-none -z-10 glow-aura" />
      <div className="absolute top-1/3 right-10 w-[400px] h-[300px] bg-accent-500/10 blur-[100px] rounded-full pointer-events-none -z-10" />

      {/* Hero Section */}
      <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 sm:pt-24 pb-16 text-center flex flex-col items-center">
        
        {/* Top Announcement Badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/90 border border-brand-500/30 text-brand-300 text-xs font-medium mb-8 shadow-sm">
          <Sparkles className="w-3.5 h-3.5 text-brand-400 animate-spin" style={{ animationDuration: "6s" }} />
          <span>Production AI • Multi-Task Deep Learning & Clinical NLP Intelligence</span>
        </div>

        {/* Main Headline */}
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold text-white tracking-tight leading-[1.1] max-w-4xl">
          Understand Medical Questions with{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 via-teal-300 to-accent-400">
            Intelligent AI
          </span>
        </h1>

        {/* Subtitle */}
        <p className="mt-6 text-base sm:text-xl text-slate-300 max-w-2xl leading-relaxed">
          MedClass AI uses deep learning and transformer-based NLP to accurately classify natural-language inquiries by medical specialty category, clinical intention, and uncertainty estimation.
        </p>

        {/* Action Buttons */}
        <div className="mt-8 flex flex-col sm:flex-row items-center gap-4 w-full justify-center">
          <Link
            href="/classify"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-3.5 rounded-xl bg-gradient-to-r from-brand-500 to-accent-600 hover:from-brand-400 hover:to-accent-500 text-white font-semibold text-base shadow-xl shadow-brand-500/25 hover:shadow-brand-500/40 transition-all active:scale-95"
          >
            <Sparkles className="w-5 h-5" />
            Try Classifier
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/analytics"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-700/80 text-slate-200 font-semibold text-base shadow-sm transition-all"
          >
            <Activity className="w-4 h-4 text-brand-400" />
            Explore Analytics
          </Link>
          <Link
            href="/models"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-3.5 rounded-xl text-slate-400 hover:text-white font-medium text-sm transition-colors"
          >
            View Benchmark Metrics
          </Link>
        </div>

        {/* Quick interactive search input bar */}
        <div className="mt-12 w-full max-w-3xl">
          <form
            onSubmit={handleQuickSubmit}
            className="glass-panel p-2 rounded-2xl flex flex-col sm:flex-row items-center gap-2 shadow-2xl border border-slate-800"
          >
            <div className="flex items-center gap-3 w-full px-3 py-2">
              <Search className="w-5 h-5 text-slate-400 shrink-0" />
              <input
                type="text"
                value={quickInput}
                onChange={(e) => setQuickInput(e.target.value)}
                placeholder="Ask any medical question (e.g., 'What are early warning signs of diabetes?')..."
                className="w-full bg-transparent text-white text-sm focus:outline-none placeholder:text-slate-500"
              />
            </div>
            <button
              type="submit"
              className="w-full sm:w-auto px-6 py-2.5 bg-brand-600 hover:bg-brand-500 text-white rounded-xl text-sm font-semibold transition-all shrink-0 flex items-center justify-center gap-2"
            >
              Analyze
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          {/* Sample Chips */}
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs">
            <span className="text-slate-400 font-medium mr-1">Quick Demos:</span>
            {sampleQuestions.map((sq, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectSample(sq)}
                className="px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-brand-300 border border-slate-800 hover:border-brand-500/30 transition-all text-left"
              >
                {sq}
              </button>
            ))}
          </div>
        </div>

      </section>

      {/* Feature Capabilities Grid */}
      <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 border-t border-slate-800/80">
        <div className="text-center mb-12">
          <h2 className="text-xs uppercase font-bold tracking-widest text-brand-400">Core Architecture & Capabilities</h2>
          <p className="mt-2 text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Engineered for Precision Medical Question Understanding
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          
          <div className="glass-panel p-6 rounded-2xl glass-card-hover">
            <div className="w-12 h-12 rounded-xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center mb-4">
              <Layers className="w-6 h-6 text-brand-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Multi-Task Transformer</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Shared contextual BERT encoder powering dual joint heads for simultaneous category and intention classification with 98.42% test accuracy.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl glass-card-hover">
            <div className="w-12 h-12 rounded-xl bg-accent-500/10 border border-accent-500/20 flex items-center justify-center mb-4">
              <ShieldCheck className="w-6 h-6 text-accent-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Medical vs Non-Medical Gate</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              First-stage domain filter distinguishes clinical queries from out-of-domain tech, trivia, or general chatter with 100% test precision.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl glass-card-hover">
            <div className="w-12 h-12 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Confidence & Uncertainty</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Calibrated softmax probabilities with active uncertainty thresholding to flag ambiguous or underspecified medical queries gracefully.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl glass-card-hover">
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mb-4">
              <Database className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Semantic Similarity Search</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Vector retrieval indexing 8,000+ verified NIH MedQuAD questions with cosine similarity scores and trusted institutional answer snippets.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl glass-card-hover">
            <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-4">
              <AlertTriangle className="w-6 h-6 text-rose-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Emergency Red Flag Triage</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Automatic alert engine detecting acute symptoms (chest pain, stroke symptoms, respiratory distress) to route users to emergency services.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl glass-card-hover">
            <div className="w-12 h-12 rounded-xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center mb-4">
              <Stethoscope className="w-6 h-6 text-sky-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Clinical Routing Guidance</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Maps questions to relevant certified medical specialties (e.g. Cardiologist, Neurologist, Pulmonologist) and hospital departments.
            </p>
          </div>

        </div>
      </section>

      {/* Model Evolution Banner */}
      <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="glass-panel p-8 sm:p-12 rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900/90 to-slate-950/90">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-brand-400">Machine Learning Evolution</span>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white mt-2">
                From Traditional NLP to Contextual Transformers
              </h2>
              <p className="mt-4 text-slate-300 text-sm leading-relaxed">
                We engineered, trained, and benchmarked three distinct model generations on the curated NIH MedQuAD corpus:
              </p>
              <ul className="mt-6 space-y-3 text-sm text-slate-300">
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-brand-400 shrink-0" />
                  <span><strong>Baseline ML:</strong> TF-IDF (10,000 n-grams) + Calibrated Linear SVM (97.97% Category Acc)</span>
                </li>
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-brand-400 shrink-0" />
                  <span><strong>Deep Learning:</strong> 2-Layer Bidirectional LSTM with pooling (94.46% Category Acc, 99.77% Intent Acc)</span>
                </li>
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-brand-400 shrink-0" />
                  <span><strong>Transformer SOTA:</strong> Contextual Multi-Task BERT (98.42% Category Acc, 99.66% Intent Acc)</span>
                </li>
              </ul>
              <div className="mt-8">
                <Link
                  href="/models"
                  className="inline-flex items-center gap-2 text-sm font-semibold text-brand-400 hover:text-brand-300 transition-colors"
                >
                  Explore Benchmark Graphs & Confusion Matrices →
                </Link>
              </div>
            </div>

            {/* Quick Metrics Comparison Card */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-center">
                <p className="text-xs text-slate-400 uppercase">Baseline SVM</p>
                <p className="text-2xl font-bold text-white mt-1">97.97%</p>
                <p className="text-[11px] text-slate-500 mt-1">Category Acc</p>
              </div>
              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-center">
                <p className="text-xs text-slate-400 uppercase">BiLSTM DL</p>
                <p className="text-2xl font-bold text-white mt-1">94.46%</p>
                <p className="text-[11px] text-slate-500 mt-1">Category Acc</p>
              </div>
              <div className="p-4 rounded-xl bg-brand-950/50 border border-brand-500/40 text-center shadow-lg shadow-brand-500/10">
                <p className="text-xs text-brand-300 uppercase font-semibold">Transformer</p>
                <p className="text-2xl font-bold text-brand-400 mt-1">98.42%</p>
                <p className="text-[11px] text-brand-300/70 mt-1">Category Acc</p>
              </div>
              
              <div className="col-span-3 p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between text-xs text-slate-400">
                <span>Multi-Task Intent Accuracy:</span>
                <span className="text-white font-bold text-sm">99.66% (Transformer SOTA)</span>
              </div>
            </div>

          </div>
        </div>
      </section>

    </div>
  );
}
