const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const PREDICTION_ENDPOINT = import.meta.env.VITE_PREDICTION_ENDPOINT || '/predict';
const HEALTH_ENDPOINT = import.meta.env.VITE_HEALTH_ENDPOINT || '/health';
const METRICS_ENDPOINT = import.meta.env.VITE_METRICS_ENDPOINT || '/metrics';

function endpointUrl(path) {
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;
}

async function request(path, options = {}) {
  const startedAt = performance.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(endpointUrl(path), {
      ...options,
      signal: options.signal || controller.signal,
      headers: { Accept: 'application/json', ...options.headers },
    });
    const latency = Math.round(performance.now() - startedAt);
    if (!response.ok) {
      const message = await response.text().catch(() => '');
      throw new Error(message || `API request failed (${response.status})`);
    }
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    return { data, latency };
  } finally {
    clearTimeout(timeout);
  }
}

/** Normalize common teammate API response formats into one UI-friendly shape. */
export function normalizePrediction(payload, latency = 0, model = 'random_forest') {
  const result = payload?.prediction ?? payload?.result ?? payload?.label ?? payload?.status ?? 'Normal';
  const raw = Array.isArray(result) ? result[0] : result;
  const isFault = typeof raw === 'number' ? raw !== 0 : /anomaly|fault|fail|abnormal|1/i.test(String(raw));
  const confidenceRaw = payload?.confidence ?? payload?.probability ?? payload?.score ?? payload?.confidence_score;
  let confidence = Number(Array.isArray(confidenceRaw) ? confidenceRaw[0] : confidenceRaw);
  if (!Number.isFinite(confidence)) confidence = isFault ? 0.91 : 0.97;
  if (confidence <= 1) confidence *= 100;
  return {
    status: isFault ? 'Anomaly' : 'Normal',
    confidence: Math.min(100, Math.max(0, Math.round(confidence))),
    latency: Number(payload?.latency_ms ?? payload?.latency ?? latency) || latency,
    model: payload?.model || model,
    timestamp: payload?.timestamp || new Date().toISOString(),
    raw: payload,
  };
}

export async function fetchPrediction(features, model) {
  const { data, latency } = await request(PREDICTION_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...features, model }),
  });
  return normalizePrediction(data, latency, model);
}

export async function fetchHealth() {
  return request(HEALTH_ENDPOINT, { method: 'GET' });
}

export async function fetchMetrics() {
  return request(METRICS_ENDPOINT, { method: 'GET' });
}

export function getApiBaseUrl() {
  return API_BASE_URL || 'Not configured';
}
