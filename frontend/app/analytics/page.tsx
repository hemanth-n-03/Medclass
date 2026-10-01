"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart3,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Flame,
  HelpCircle,
  TrendingUp,
  RefreshCw,
  Clock
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell
} from "recharts";
import { fetchAnalytics, fetchModelInfo } from "@/lib/api";
import { AnalyticsSummary, ModelInfo } from "@/types";

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [ana, minfo] = await Promise.all([
        fetchAnalytics(),
        fetchModelInfo().catch(() => null),
      ]);
      setAnalytics(ana);
      setModelInfo(minfo);
    } catch (err: any) {
      setError(err.message || "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Format category distribution for Recharts
  const categoryChartData = analytics
    ? Object.entries(analytics.category_distribution)
        .filter(([cat]) => cat !== "Non-Medical")
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, count]) => ({
          name: name.length > 14 ? name.substring(0, 12) + ".." : name,
          fullName: name,
          queries: count,
        }))
    : [];

  // Format intent distribution for Recharts
  const intentChartData = analytics
    ? Object.entries(analytics.intent_distribution)
        .filter(([i]) => i !== "Non-Medical")
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => ({
          name,
          count,
        }))
    : [];

  // Colors for chart bars
  const BAR_COLORS = [
    "#14b8a6", "#2dd4bf", "#5eead4", "#6366f1", "#818cf8",
    "#a5b4fc", "#38bdf8", "#0284c7", "#ec4899", "#f43f5e"
  ];

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-semibold mb-2">
            <BarChart3 className="w-3.5 h-3.5" />
            Live Clinical Analytics & System Auditing
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Analytics & Model Intelligence Dashboard
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Real-time query metrics, class distributions, uncertainty audits, and model benchmarks.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white text-xs font-semibold transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh Metrics
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-200 text-sm mb-6 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {/* KPI Cards */}
      {analytics && (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          
          <div className="glass-panel p-5 rounded-2xl border border-slate-800">
            <span className="text-[11px] font-semibold uppercase text-slate-400 block">Total Inquiries</span>
            <p className="text-2xl font-extrabold text-white mt-1">
              {analytics.total_queries.toLocaleString()}
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">Trained & Live Events</span>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-slate-800">
            <span className="text-[11px] font-semibold uppercase text-brand-400 block">Clinical Queries</span>
            <p className="text-2xl font-extrabold text-brand-300 mt-1">
              {analytics.medical_queries.toLocaleString()}
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">
              {((analytics.medical_queries / Math.max(1, analytics.total_queries)) * 100).toFixed(1)}% of total
            </span>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-slate-800">
            <span className="text-[11px] font-semibold uppercase text-accent-400 block">Avg. Confidence</span>
            <p className="text-2xl font-extrabold text-accent-300 mt-1">
              {(analytics.avg_confidence * 100).toFixed(1)}%
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">Softmax probability</span>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-slate-800">
            <span className="text-[11px] font-semibold uppercase text-amber-400 block">Uncertainty Flagged</span>
            <p className="text-2xl font-extrabold text-amber-300 mt-1">
              {analytics.uncertain_queries.toLocaleString()}
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">
              {((analytics.uncertain_queries / Math.max(1, analytics.total_queries)) * 100).toFixed(2)}% flagged
            </span>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-slate-800 col-span-2 lg:col-span-1">
            <span className="text-[11px] font-semibold uppercase text-rose-400 block">Emergency Alerts</span>
            <p className="text-2xl font-extrabold text-rose-300 mt-1">
              {analytics.emergency_queries.toLocaleString()}
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">Safety triage triggers</span>
          </div>

        </div>
      )}

      {/* Distribution Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        
        {/* Category Distribution Chart */}
        <div className="glass-panel p-6 rounded-3xl border border-slate-800 flex flex-col">
          <div className="mb-4">
            <h3 className="text-base font-bold text-white">Top Medical Category Distribution</h3>
            <p className="text-xs text-slate-400">Class frequency across NIH MedQuAD corpus & audit logs</p>
          </div>
          <div className="h-72 w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryChartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={10}
                  angle={-30}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "10px", fontSize: "12px" }}
                  formatter={(val: any, name: any, item: any) => [val, item.payload.fullName]}
                />
                <Bar dataKey="queries" radius={[4, 4, 0, 0]}>
                  {categoryChartData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Intent Distribution Chart */}
        <div className="glass-panel p-6 rounded-3xl border border-slate-800 flex flex-col">
          <div className="mb-4">
            <h3 className="text-base font-bold text-white">Clinical Intention Distribution</h3>
            <p className="text-xs text-slate-400">Distribution of question communicative intent</p>
          </div>
          <div className="h-72 w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={intentChartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={10}
                  angle={-30}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "10px", fontSize: "12px" }}
                />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Model Benchmark Comparison Graph */}
      {modelInfo?.benchmarks?.chart_data && (
        <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-800 mb-8">
          <div className="mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-brand-400">Model Verification</span>
              <h3 className="text-xl font-bold text-white mt-1">
                Comparative Test Set Performance (% Accuracy & Macro F1)
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Measured on the exact identical held-out test split (957 records) across architectures from the expanded 9,566 dataset.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5 text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-400" /> Baseline SVM
              </span>
              <span className="flex items-center gap-1.5 text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-400" /> BiLSTM Deep
              </span>
              <span className="flex items-center gap-1.5 text-brand-300 font-bold">
                <span className="w-2.5 h-2.5 rounded-full bg-brand-400" /> Transformer SOTA
              </span>
            </div>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelInfo.benchmarks.chart_data} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="task" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={[80, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "10px", fontSize: "12px" }}
                  formatter={(val: any) => [`${Number(val).toFixed(2)}%`]}
                />
                <Legend wrapperStyle={{ paddingTop: "10px", fontSize: "12px" }} />
                <Bar dataKey="Baseline_SVM" name="Baseline (Linear SVM)" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="BiLSTM" name="Deep Learning (BiLSTM)" fill="#818cf8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Transformer" name="Contextual Transformer" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Recent Query Stream Table */}
      {analytics?.recent_activity && analytics.recent_activity.length > 0 && (
        <div className="glass-panel p-6 rounded-3xl border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Clock className="w-4 h-4 text-brand-400" />
              Live Query Audit Stream
            </h3>
            <span className="text-xs text-slate-500">Last 15 requests</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                  <th className="py-3 px-4">Time (IST)</th>
                  <th className="py-3 px-4">Inquiry Snippet</th>
                  <th className="py-3 px-4">Domain</th>
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Intent</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Flags</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {analytics.recent_activity.map((evt, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-3 px-4 text-slate-400 font-mono text-[11px] whitespace-nowrap">
                      {evt.timestamp_ist || (evt.timestamp ? evt.timestamp.substring(11, 19) + " UTC" : "Just now")}
                    </td>
                    <td className="py-3 px-4 text-slate-200 font-medium max-w-xs truncate">
                      {evt.question_preview}
                    </td>
                    <td className="py-3 px-4">
                      {evt.is_medical ? (
                        <span className="px-2 py-0.5 rounded bg-brand-500/10 text-brand-300 font-medium border border-brand-500/20">
                          Medical
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                          Non-Medical
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-slate-300">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span>{evt.primary_category || evt.category || "—"}</span>
                        {evt.is_multi_symptom && (
                          <span className="px-1.5 py-0.2 rounded bg-teal-950 text-teal-300 border border-teal-500/30 text-[10px]">
                            Multi
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-slate-300">{evt.intent || "—"}</td>
                    <td className="py-3 px-4 font-mono text-slate-200">
                      {(evt.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 flex items-center gap-1.5 flex-wrap">
                      {evt.emergency && (
                        <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold border border-rose-500/30">
                          CRITICAL
                        </span>
                      )}
                      {!evt.emergency && evt.safety_level === "urgent" && (
                        <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-medium border border-amber-500/30">
                          URGENT
                        </span>
                      )}
                      {evt.uncertain && (
                        <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Uncertain
                        </span>
                      )}
                      {!evt.emergency && evt.safety_level !== "urgent" && !evt.uncertain && (
                        <span className="text-slate-500">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}
