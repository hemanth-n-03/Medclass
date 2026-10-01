import React from "react";
import Link from "next/link";
import {
  ShieldAlert,
  ShieldCheck,
  HeartHandshake,
  Lock,
  AlertOctagon,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Sparkles,
  PhoneCall
} from "lucide-react";

export default function AboutPage() {
  return (
    <div className="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      
      {/* Title */}
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs font-semibold mb-3">
          <ShieldAlert className="w-3.5 h-3.5" />
          Responsible AI & Clinical Safety Framework
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          About MedClass AI & Safety Ethics
        </h1>
        <p className="mt-2 text-sm sm:text-base text-slate-400">
          Transparent guidelines on intended use, non-diagnostic boundaries, uncertainty disclosure, and user safety safeguards.
        </p>
      </div>

      {/* Emergency Hotline Notice */}
      <div className="p-6 rounded-3xl bg-rose-950/40 border-2 border-rose-500/50 mb-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center shrink-0">
            <PhoneCall className="w-6 h-6 text-rose-400" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Experiencing an Acute Emergency?</h3>
            <p className="text-xs text-rose-200/90 mt-0.5">
              Do not use this application for emergency medical needs. Call your local emergency hotline immediately:
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs font-bold text-white shrink-0 flex-wrap">
          <span className="px-3 py-1.5 rounded-lg bg-rose-600 flex items-center gap-1.5">
            <PhoneCall className="w-3.5 h-3.5" />
            India Emergency: 112
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-rose-700 flex items-center gap-1.5">
            <PhoneCall className="w-3.5 h-3.5" />
            Ambulance Service: 108
          </span>
        </div>
      </div>

      {/* Boundary Matrix: What MedClass AI Does vs What It Does NOT Do */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        
        {/* WHAT IT DOES */}
        <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-emerald-500/30">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <h3 className="text-lg font-bold text-white">What the System DOES</h3>
          </div>
          <ul className="space-y-3 text-xs sm:text-sm text-slate-300">
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
              <span><strong>Classifies the Question:</strong> Identifies whether a written question is clinical or non-clinical in nature.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
              <span><strong>Taxonomy Routing:</strong> Categorizes inquiry topics into 13 medical specialties (e.g. Cardiology, Endocrinology).</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
              <span><strong>Intention Extraction:</strong> Detects user intent (Symptoms, Causes, Diagnosis, Treatment, Prevention, Side Effects).</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
              <span><strong>Semantic Retrieval:</strong> Surfaces similar verified inquiries from the NIH MedQuAD reference repository.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
              <span><strong>Uncertainty Disclosure:</strong> Transparently highlights predictions with low confidence or high semantic entropy.</span>
            </li>
          </ul>
        </div>

        {/* WHAT IT DOES NOT DO */}
        <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-rose-500/30">
          <div className="flex items-center gap-2 mb-4">
            <XCircle className="w-5 h-5 text-rose-400" />
            <h3 className="text-lg font-bold text-white">What the System NEVER Does</h3>
          </div>
          <ul className="space-y-3 text-xs sm:text-sm text-slate-300">
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
              <span><strong>Never Diagnoses:</strong> Does NOT provide clinical diagnosis or tell a user what medical condition they have.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
              <span><strong>Never Prescribes:</strong> Does NOT prescribe medications, adjust drug doses, or recommend off-label pharmaceutical use.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
              <span><strong>Never Replaces Clinicians:</strong> Is NOT a substitute for face-to-face evaluation by board-certified healthcare professionals.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
              <span><strong>Never Generates Unverified Claims:</strong> Does not hallucinate medical facts; strictly uses grounded NIH reference data.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
              <span><strong>Never Misrepresents Certainty:</strong> Does not fabricate false confidence scores or overstate predictive certainty.</span>
            </li>
          </ul>
        </div>

      </div>

      {/* Core Safety Principles */}
      <div className="glass-panel p-8 rounded-3xl border border-slate-800 mb-12">
        <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-brand-400" />
          Responsible AI Engineering Principles
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs leading-relaxed text-slate-300">
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800">
            <div className="font-bold text-white text-sm mb-2 flex items-center gap-1.5">
              <HeartHandshake className="w-4 h-4 text-brand-400" />
              Patient-First Triage
            </div>
            Questions mentioning emergency keywords (e.g. crushing chest pain, slurred speech, anaphylaxis) trigger an immediate high-priority warning redirecting the user to emergency services.
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800">
            <div className="font-bold text-white text-sm mb-2 flex items-center gap-1.5">
              <HelpCircle className="w-4 h-4 text-accent-400" />
              Epistemic Humility
            </div>
            When a question is ambiguous or underspecified (such as single-word queries like 'diabetes?'), the model signals uncertainty and requests further clinical context rather than making overconfident assumptions.
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800">
            <div className="font-bold text-white text-sm mb-2 flex items-center gap-1.5">
              <Lock className="w-4 h-4 text-teal-400" />
              Data Privacy by Design
            </div>
            Inquiries are processed locally without storing personally identifiable health information (PHI) or associating queries with user identities.
          </div>
        </div>
      </div>

      {/* Dataset & Model Limitations */}
      <div className="glass-panel p-8 rounded-3xl border border-slate-800 text-xs text-slate-300 leading-relaxed space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <AlertOctagon className="w-5 h-5 text-amber-400" />
          Known Limitations & Technical Disclosures
        </h3>
        <p>
          <strong>Dataset Representation:</strong> MedClass AI is trained on English-language clinical questions from the NIH MedQuAD corpus and curated open-domain inquiries. It is not currently optimized for non-English questions, non-standard medical vernacular, or pediatric slang.
        </p>
        <p>
          <strong>Class Imbalance:</strong> While balanced sampling was implemented, conditions in ultra-rare diseases may exhibit higher predictive variance compared to common chronic conditions like diabetes or hypertension.
        </p>
        <p>
          <strong>Clinical Verification:</strong> All predictions and routing suggestions are advisory and provided solely for clinical education, research exploration, and clinical inquiry routing.
        </p>
      </div>

    </div>
  );
}
