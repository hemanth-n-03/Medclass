import { ClassifyResponse, HealthResponse, AnalyticsSummary, ModelInfo } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";

export async function classifyQuestion(question: string): Promise<ClassifyResponse> {
  const res = await fetch(`${API_BASE_URL}/classify`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Classification request failed (${res.status})`);
  }

  return res.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function fetchAnalytics(): Promise<AnalyticsSummary> {
  const res = await fetch(`${API_BASE_URL}/analytics`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch analytics (${res.status})`);
  }
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE_URL}/model-info`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch model info (${res.status})`);
  }
  return res.json();
}
