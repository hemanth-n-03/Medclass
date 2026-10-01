"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Brain,
  Sparkles,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Stethoscope,
  BookOpen,
  Layers,
  ChevronRight,
  ShieldAlert,
  ArrowRight,
  Info,
  PhoneCall,
  Activity,
  ListOrdered,
  Clock,
  Compass
} from "lucide-react";
import { classifyQuestion } from "@/lib/api";
import { ClassifyResponse } from "@/types";

function ClassifierContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";

  const [question, setQuestion] = useState(initialQuery);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadingSteps = [
    "Analyzing natural-language query...",
    "Scanning multi-symptom clinical entities & signals...",
    "Evaluating acute emergency combinations (India 112 / 108)...",
    "Running multi-task transformer inference & routing...",
  ];

  const examplePresets = [
    {
      label: "Conversational Thirst & Urination",
      text: "I have been thirsty all day and going to the bathroom constantly, what could be wrong?"
    },
    {
      label: "Misspelled Inquiry (diabetis)",
      text: "wat r the signs of diabetis"
    },
    {
      label: "Multi-Symptom Composite",
      text: "I am having leg pain, fever and chest pain"
    },
    {
      label: "Emergency Red-Flag (Chest + Dyspnea)",
      text: "I have severe chest pain and difficulty breathing"
    },
    {
      label: "Informal Migraine & Light",
      text: "head hurts one side and light makes it worse"
    },
    {
      label: "Heart Palpitations",
      text: "heart feels like it is skipping beats when I lie down"
    },
    {
      label: "Shortness of Breath",
      text: "can't catch my breath after walking up stairs"
    },
    {
      label: "Skin Rash & Pruritus",
      text: "my skin has red patches that itch like crazy"
    },
    {
      label: "Pediatric Malaise",
      text: "kid has high temp and won't eat anything"
    },
    {
      label: "Non-Medical Domain Gate",
      text: "how do I reset my wifi router"
    }
  ];

  // Auto classify if query param present on load
  useEffect(() => {
    if (initialQuery && !result && !loading) {
      handleClassify(initialQuery);
    }
  }, [initialQuery]);

  const handleClassify = async (textToClassify: string) => {
    const q = textToClassify.trim();
    if (!q || q.length < 2) {
      setError("Please enter a question with at least 2 characters.");
      return;
    }

    setError(null);
    setLoading(true);
    setLoadingStep(0);

    const stepInterval = setInterval(() => {
      setLoadingStep((prev) => (prev < loadingSteps.length - 1 ? prev + 1 : prev));
    }, 280);

    try {
      const data = await classifyQuestion(q);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to classify question. Please check backend connection.");
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion("");
    setResult(null);
    setError(null);
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-semibold mb-3">
          <Brain className="w-3.5 h-3.5" />
          Multi-Task Clinical NLP Inference & Triage
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Medical Question Classifier
        </h1>
        <p className="mt-2 text-sm sm:text-base text-slate-400">
          Submit any health inquiry to classify its medical domain, multi-symptom signals, clinical specialty, and intention.
        </p>
      </div>

      {/* Input Card */}
      <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-800 shadow-2xl max-w-4xl mx-auto mb-10">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleClassify(question);
          }}
        >
          <div className="relative">
            <textarea
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                if (error) setError(null);
              }}
              rows={3}
              placeholder="Enter your medical question (e.g., 'I am having leg pain, fever and chest pain' or 'What are the early symptoms of diabetes?')..."
              className="w-full bg-slate-900/90 border border-slate-700/80 rounded-2xl p-4 text-white placeholder:text-slate-500 text-base focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 transition-all resize-none"
              maxLength={1000}
            />
            
            <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className={question.length > 800 ? "text-amber-400 font-medium" : ""}>
                  {question.length}
                </span>
                <span>/ 1000 characters</span>
              </span>

              {question && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="flex items-center gap-1 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Clear Input
                </button>
              )}
            </div>
          </div>

          {/* Action Row */}
          <div className="mt-5 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Info className="w-4 h-4 text-brand-400 shrink-0" />
              <span>Multi-Task Transformer • Entity Recognition • India Emergency Triage</span>
            </div>

            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-3 rounded-xl bg-gradient-to-r from-brand-500 to-accent-600 hover:from-brand-400 hover:to-accent-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm shadow-lg shadow-brand-500/25 transition-all active:scale-95"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Classify Question</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Example Chips */}
        <div className="mt-8 pt-6 border-t border-slate-800">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Clinical Test Scenarios:
          </p>
          <div className="flex flex-wrap gap-2">
            {examplePresets.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setQuestion(preset.text);
                  handleClassify(preset.text);
                }}
                className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-brand-300 border border-slate-800 hover:border-brand-500/30 text-xs font-medium transition-all"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* Loading Steps Animation */}
      {loading && (
        <div className="glass-panel p-8 rounded-3xl border border-brand-500/30 shadow-2xl max-w-2xl mx-auto my-8 text-center animate-pulse">
          <div className="w-12 h-12 rounded-2xl bg-brand-500/20 border border-brand-500/40 mx-auto flex items-center justify-center mb-4">
            <Brain className="w-6 h-6 text-brand-400 animate-bounce" />
          </div>
          <h3 className="text-lg font-bold text-white mb-2">Analyzing Clinical Inquiry</h3>
          <p className="text-sm text-brand-300 font-mono">
            {loadingSteps[loadingStep]}
          </p>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-6 overflow-hidden">
            <div
              className="bg-brand-500 h-full transition-all duration-300 rounded-full"
              style={{ width: `${((loadingStep + 1) / loadingSteps.length) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="max-w-4xl mx-auto mb-8 p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-200 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {/* Prediction Results UI */}
      {result && !loading && (
        <div className="max-w-5xl mx-auto space-y-6">

          {/* Time & Audit Metadata Badge */}
          {result.timestamp_ist && (
            <div className="flex items-center justify-between text-xs text-slate-400 px-2">
              <span className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-brand-400" />
                Query Time: <span className="font-mono text-slate-200">{result.timestamp_ist}</span>
              </span>
              <span className="text-[11px] px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                Classification & Routing Engine (Non-Diagnostic)
              </span>
            </div>
          )}

          {/* Semantic Preprocessing & Normalization Notice */}
          {result.normalized_question && result.normalized_question.toLowerCase() !== result.question.toLowerCase() && (
            <div className="p-4 rounded-2xl bg-indigo-950/40 border border-indigo-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
              <div className="space-y-1">
                <span className="font-semibold text-indigo-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  Semantic Preprocessing & Query Normalization Applied
                </span>
                <p className="text-slate-300">
                  Original: <span className="text-slate-400 italic">"{result.question}"</span>
                  {" → "}
                  Normalized: <strong className="text-indigo-200">"{result.normalized_question}"</strong>
                </p>
              </div>
              {result.corrections && result.corrections.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {result.corrections.map((c, i) => (
                    <span key={i} className="px-2 py-0.5 rounded bg-indigo-900/60 border border-indigo-500/40 text-indigo-200 font-mono text-[11px]">
                      {c.original} → {c.replacement || "removed"}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Critical Emergency Banner (Indian Helpline 112 / 108) */}
          {result.emergency?.is_emergency && (
            <div className="p-6 rounded-3xl bg-rose-950/70 border-2 border-rose-500 shadow-2xl shadow-rose-950/60 flex flex-col sm:flex-row items-start sm:items-center gap-5 text-rose-100">
              <div className="w-14 h-14 rounded-2xl bg-rose-500/20 border border-rose-500 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-8 h-8 text-rose-400 animate-pulse" />
              </div>
              <div className="space-y-2 flex-1">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <h3 className="text-lg font-bold text-rose-300 flex items-center gap-2">
                    Critical Red-Flag Medical Indicators Detected
                  </h3>
                  <span className="px-3 py-1 rounded-full bg-rose-500 text-white text-xs font-bold uppercase tracking-wider animate-pulse">
                    Immediate Action Required
                  </span>
                </div>
                <p className="text-xs sm:text-sm text-rose-200 leading-relaxed">
                  {result.emergency.warning}
                </p>
                <div className="pt-2 flex flex-wrap items-center gap-3">
                  <a
                    href="tel:112"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs shadow-md"
                  >
                    <PhoneCall className="w-4 h-4" />
                    Call 112 (National Helpline)
                  </a>
                  <a
                    href="tel:108"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-700 hover:bg-rose-600 text-white font-bold text-xs shadow-md"
                  >
                    <PhoneCall className="w-4 h-4" />
                    Call 108 (Ambulance)
                  </a>
                  <span className="text-xs text-rose-300">
                    Or proceed to the nearest hospital emergency department.
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Urgent Symptom Warning (e.g. Chest pain alone, or high-acuity multi-symptom) */}
          {!result.emergency?.is_emergency && result.safety_level === "urgent" && (
            <div className="p-5 rounded-2xl bg-amber-950/50 border border-amber-500/60 flex items-start sm:items-center gap-4 text-amber-100">
              <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              </div>
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-amber-300">
                  Urgent In-Person Assessment Recommended
                </h4>
                <p className="text-xs text-amber-200/90 leading-relaxed">
                  {result.emergency?.warning || "Symptoms in your inquiry warrant prompt evaluation by a registered medical practitioner. Seek emergency care immediately if breathlessness or dizziness develops."}
                </p>
              </div>
            </div>
          )}

          {/* Non-Medical Domain Banner (OOD) */}
          {!result.is_medical ? (
            <div className="glass-panel p-8 rounded-3xl border border-amber-500/30 text-center space-y-4">
              <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/30 mx-auto flex items-center justify-center">
                <HelpCircle className="w-8 h-8 text-amber-400" />
              </div>
              <span className="inline-block px-3 py-1 rounded-full bg-amber-500/20 text-amber-300 text-xs font-semibold">
                Domain Gate: Non-Medical ({(result.domain_confidence * 100).toFixed(1)}% confidence)
              </span>
              <h2 className="text-xl sm:text-2xl font-bold text-white">
                Outside Supported Medical Domain
              </h2>
              <p className="text-slate-300 text-sm max-w-xl mx-auto">
                {result.message || "This question appears to be outside the supported medical domain."}
              </p>
              <p className="text-xs text-slate-500">
                MedClass AI classifies clinical inquiries and symptom profiles. Please ask a health, disease, medication, or biomedical question.
              </p>
            </div>
          ) : (
            <>
              {/* Multi-Symptom / Multi-Entity Signal Recognition Panel */}
              {(result.is_multi_symptom || (result.detected_entities && result.detected_entities.length > 0)) && (
                <div className="glass-panel p-6 sm:p-7 rounded-3xl border border-teal-500/30">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
                    <span className="text-xs font-bold uppercase tracking-wider text-teal-400 flex items-center gap-1.5">
                      <Activity className="w-4 h-4" />
                      Detected Health Concerns & Clinical Signals
                    </span>
                    {result.is_multi_symptom && (
                      <span className="px-2.5 py-0.5 rounded-full bg-teal-500/20 text-teal-300 text-xs font-bold border border-teal-500/30 self-start sm:self-auto">
                        Composite Multi-Symptom Query
                      </span>
                    )}
                  </div>

                  {/* Badges of detected concerns */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {result.detected_entities.map((concern, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 rounded-xl bg-slate-900 border border-slate-700/80 text-white font-medium text-xs flex items-center gap-1.5 shadow-sm"
                      >
                        <span className="w-2 h-2 rounded-full bg-teal-400" />
                        {concern}
                      </span>
                    ))}
                  </div>

                  {/* Detailed Clinical Signal narratives */}
                  {result.clinical_signals && result.clinical_signals.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                      {result.clinical_signals.map((sig, idx) => (
                        <div
                          key={idx}
                          className="p-3.5 rounded-2xl bg-slate-900/70 border border-slate-800 text-xs space-y-1.5"
                        >
                          <div className="flex items-center justify-between text-slate-300 font-semibold">
                            <span>{sig.entity}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-500/30">
                              {sig.domain}
                            </span>
                          </div>
                          <p className="text-slate-400 text-[11px] leading-relaxed">
                            {sig.signal}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Primary Medical Specialty & Intent Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* PRIMARY CATEGORY CARD */}
                <div className="glass-panel p-6 sm:p-7 rounded-3xl border border-brand-500/30 relative overflow-hidden">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs font-bold uppercase tracking-wider text-brand-400 flex items-center gap-1.5">
                      <Stethoscope className="w-4 h-4" />
                      Primary Clinical Specialty
                    </span>
                    <div className="text-right">
                      <span className="px-2.5 py-1 rounded-full bg-brand-500/20 text-brand-300 text-xs font-bold border border-brand-500/30">
                        {((result.primary_category_confidence || result.category_confidence) * 100).toFixed(1)}% Confidence
                      </span>
                      <span className="block text-[10px] text-slate-400 mt-1">
                        Model Confidence (Not Medical Certainty)
                      </span>
                    </div>
                  </div>

                  <h3 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                    {result.primary_category || result.category}
                  </h3>

                  {/* Progress bar */}
                  <div className="w-full bg-slate-800 h-2 rounded-full my-4 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-brand-500 to-teal-400 h-full rounded-full transition-all duration-700"
                      style={{ width: `${(result.primary_category_confidence || result.category_confidence) * 100}%` }}
                    />
                  </div>

                  {/* Primary Selection Rationale */}
                  {result.primary_category_reason && (
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300 mb-4 leading-relaxed">
                      <strong className="text-brand-300 font-semibold block mb-0.5">Classification Rationale:</strong>
                      {result.primary_category_reason}
                    </div>
                  )}

                  {/* Secondary Categories with Rationale */}
                  {result.secondary_categories && result.secondary_categories.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                      <span className="text-slate-400 font-medium block mb-2">Secondary Clinical Domains:</span>
                      <div className="space-y-1.5">
                        {result.secondary_categories.map((sub, i) => (
                          <div
                            key={i}
                            className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-start justify-between gap-2 text-[11px]"
                          >
                            <div>
                              <span className="font-semibold text-white">{sub.category}</span>
                              {sub.reason && (
                                <p className="text-slate-400 text-[10px] mt-0.5 leading-snug">{sub.reason}</p>
                              )}
                            </div>
                            <span className="font-mono text-brand-300 font-bold shrink-0">
                              {(sub.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* CLINICAL INTENTION CARD */}
                <div className="glass-panel p-6 sm:p-7 rounded-3xl border border-accent-500/30 relative overflow-hidden">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs font-bold uppercase tracking-wider text-accent-400 flex items-center gap-1.5">
                      <BookOpen className="w-4 h-4" />
                      Inquiry Intention
                    </span>
                    <span className="px-2.5 py-1 rounded-full bg-accent-500/20 text-accent-300 text-xs font-bold border border-accent-500/30">
                      {(result.intent_confidence * 100).toFixed(1)}% Confidence
                    </span>
                  </div>

                  <h3 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                    {result.intent}
                  </h3>

                  {/* Progress bar */}
                  <div className="w-full bg-slate-800 h-2 rounded-full my-4 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-accent-500 to-purple-400 h-full rounded-full transition-all duration-700"
                      style={{ width: `${result.intent_confidence * 100}%` }}
                    />
                  </div>

                  {/* Secondary Intents */}
                  {result.secondary_intents && result.secondary_intents.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                      <span className="text-slate-400 font-medium block mb-2">Secondary Intentions:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {result.secondary_intents.map((sInt, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded bg-slate-800/80 text-slate-300 border border-slate-700/60"
                          >
                            {sInt.intent} ({(sInt.confidence * 100).toFixed(0)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Uncertainty Banner (if flagged) */}
                  {result.uncertain && (
                    <div className="mt-4 p-3.5 rounded-2xl bg-amber-950/40 border border-amber-500/40 flex items-start gap-2.5 text-amber-200 text-xs leading-relaxed">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      <div>
                        <strong className="font-semibold text-amber-300 block mb-0.5">High Uncertainty Notice:</strong>
                        {result.uncertainty_reason}
                      </div>
                    </div>
                  )}

                  {/* Ambiguity Clarification Prompt (for single words like 'Headache') */}
                  {result.clarification_needed && (
                    <div className="mt-4 p-4 rounded-2xl bg-sky-950/40 border border-sky-500/40 text-sky-200 text-xs space-y-2">
                      <div className="flex items-center gap-2 font-bold text-sky-300">
                        <HelpCircle className="w-4 h-4 text-sky-400" />
                        Clarification Recommended for Accurate Classification
                      </div>
                      <p className="leading-relaxed text-sky-100">
                        {result.clarification_needed}
                      </p>
                    </div>
                  )}

                </div>

              </div>

              {/* Suggested Next Steps (Proportional, Non-Diagnostic) */}
              {result.suggested_next_steps && result.suggested_next_steps.length > 0 && (
                <div className="glass-panel p-6 sm:p-7 rounded-3xl border border-slate-800">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
                    <ListOrdered className="w-4 h-4 text-brand-400" />
                    Proportional Next Steps (Non-Diagnostic Guidance)
                  </h4>
                  <ul className="space-y-2 text-xs text-slate-300">
                    {result.suggested_next_steps.map((step, idx) => (
                      <li key={idx} className="flex items-start gap-2.5">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                        <span className="leading-relaxed">{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Explainability & Clinical Routing Section */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Explainable AI */}
                {result.explanation && (
                  <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-teal-400 mb-2 flex items-center gap-2">
                      <Layers className="w-4 h-4" />
                      Model Feature Attribution & Explanation
                    </h4>
                    <p className="text-xs text-slate-300 mb-4 leading-relaxed">
                      {result.explanation.explanation_text}
                    </p>
                    <div>
                      <span className="text-[11px] text-slate-400 block mb-1.5 font-medium">Salient Entity Tokens:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {result.explanation.salient_tokens.map((token, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded-md bg-teal-950/60 border border-teal-500/30 text-teal-300 font-mono text-xs"
                          >
                            {token}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Routing & Guidance */}
                {result.routing && (
                  <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-sky-400 mb-2 flex items-center gap-2">
                      <Compass className="w-4 h-4" />
                      Clinical Referral & Routing Suggestion
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div>
                        <span className="text-slate-400">Recommended Practitioner: </span>
                        <strong className="text-white text-sm">{result.routing.recommended_specialist}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400">Department: </span>
                        <span className="text-slate-200">{result.routing.clinical_department}</span>
                      </div>
                      <p className="text-slate-400 pt-1 leading-relaxed">
                        {result.routing.suggested_action}
                      </p>
                    </div>
                  </div>
                )}

              </div>

              {/* Similar Questions from NIH Knowledge Base */}
              {result.similar_questions && result.similar_questions.length > 0 && (
                <div className="glass-panel p-6 sm:p-7 rounded-3xl border border-slate-800">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-sm font-bold text-white flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-brand-400" />
                      Similar Questions from Verified NIH Knowledge Base ({result.similar_questions.length})
                    </h4>
                    <span className="text-xs text-slate-400">Semantic & Intent Reranked</span>
                  </div>

                  <div className="space-y-3">
                    {result.similar_questions.map((sim, i) => (
                      <div
                        key={i}
                        className="p-4 rounded-xl bg-slate-900/70 border border-slate-800/80 hover:border-slate-700 transition-all text-xs"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <p className="font-semibold text-slate-100 text-sm">
                            {sim.question}
                          </p>
                          <span className="px-2 py-0.5 rounded bg-brand-950 border border-brand-500/30 text-brand-300 font-mono text-[11px] shrink-0">
                            {(sim.similarity_score * 100).toFixed(0)}% Match
                          </span>
                        </div>
                        {sim.answer_snippet && (
                          <p className="mt-2 text-slate-400 leading-relaxed line-clamp-2">
                            {sim.answer_snippet}
                          </p>
                        )}
                        <div className="mt-2.5 flex items-center gap-3 text-[11px] text-slate-500">
                          <span>
                            Source: {sim.source_organization || sim.source || "NIH"}
                          </span>
                          {sim.focus_area && <span>Focus: {sim.focus_area}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Responsible AI Disclaimer Card */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center gap-3 text-slate-400 text-xs">
                <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0" />
                <p>{result.safety_disclaimer}</p>
              </div>

            </>
          )}

        </div>
      )}

    </div>
  );
}

export default function ClassifierPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-slate-400">Loading Classifier...</div>}>
      <ClassifierContent />
    </Suspense>
  );
}
