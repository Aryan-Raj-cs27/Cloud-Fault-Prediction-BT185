// All requests are same-origin under /api (Vite proxy in dev, FastAPI in production).
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function request(path, options = {}) {
  const startedAt = performance.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      credentials: 'same-origin',
      signal: controller.signal,
      headers: { Accept: 'application/json', ...options.headers },
    });
    const latency = Math.round(performance.now() - startedAt);
    if (!response.ok) {
      let message = '';
      try { message = (await response.json())?.detail; } catch { /* non-JSON error body */ }
      if (response.status === 401 && !path.startsWith('/auth/')) window.dispatchEvent(new Event('fa:unauthorized'));
      throw new ApiError(typeof message === 'string' && message ? message : `Request failed (${response.status})`, response.status);
    }
    return { data: await response.json(), latency };
  } catch (error) {
    if (error.name === 'AbortError') throw new ApiError('The server did not answer within 15 s.', 0);
    if (error instanceof ApiError) throw error;
    throw new ApiError('Cannot reach the server.', 0);
  } finally {
    clearTimeout(timeout);
  }
}

const postJson = (path, body) => request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const WINDOWS = {
  w60: { label: 'Early warning · 60 s', short: '60 s' },
  w10: { label: 'Very early · 10 s', short: '10 s' },
  full: { label: 'Full session (detection)', short: 'Full' },
};

/** Map the /predict response into the UI record shape. */
export function normalizePrediction(payload, latency = 0, truth = null, blockId = null) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    status: payload.prediction === 1 ? 'Anomaly' : 'Normal',
    risk: Math.round(payload.risk * 1000) / 10,
    riskLevel: payload.risk_level,
    latency: Number(payload.latency_ms ?? latency),
    window: payload.window,
    truth,
    blockId,
    timestamp: new Date().toISOString(),
    raw: payload,
  };
}

export async function fetchPrediction(logLines, window, truth = null, blockId = null) {
  const { data, latency } = await postJson('/predict', { log_lines: logLines, window });
  return normalizePrediction(data, latency, truth, blockId);
}

export const login = (username, password) => postJson('/auth/login', { username, password });
export const logout = () => request('/auth/logout', { method: 'POST' });
export const fetchMe = () => request('/auth/me');
export const fetchHealth = () => request('/health');
export const fetchMetrics = (window) => request(`/metrics${window ? `?window=${encodeURIComponent(window)}` : ''}`);
export const fetchSimulation = () => request('/simulation?max_points=240');
export const fetchSample = (label) => request(`/samples/next${label ? `?label=${label}` : ''}`);

/** Group raw HDFS lines by block id (a line can mention several blocks). */
export function groupLinesByBlock(text) {
  const groups = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (line.length > 4000) continue;
    for (const block of new Set(line.match(/blk_-?\d+/g) || [])) {
      if (!groups.has(block)) groups.set(block, []);
      groups.get(block).push(line);
    }
  }
  return groups;
}

export const getApiBaseUrl = () => API_BASE_URL;
