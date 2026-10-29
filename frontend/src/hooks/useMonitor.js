import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchHealth, fetchMetrics, fetchPrediction } from '../lib/api.js';

const seedHealth = [94, 96, 92, 97, 95, 98, 93, 96, 99, 95];
const seed = seedHealth.map((health, index) => ({
  time: new Date(Date.now() - (seedHealth.length - index - 1) * 10000)
    .toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' }),
  health,
  cpu: [38, 42, 45, 39, 47, 43, 52, 48, 41, 46][index],
}));

const initialLogs = [
  { id: 'log-1', timestamp: new Date(Date.now() - 32000).toISOString(), model: 'Random Forest', status: 'Normal', confidence: 98, latency: 32 },
  { id: 'log-2', timestamp: new Date(Date.now() - 59000).toISOString(), model: 'CNN-LSTM', status: 'Normal', confidence: 95, latency: 41 },
  { id: 'log-3', timestamp: new Date(Date.now() - 91000).toISOString(), model: 'Random Forest', status: 'Anomaly', confidence: 87, latency: 29 },
  { id: 'log-4', timestamp: new Date(Date.now() - 132000).toISOString(), model: 'CNN-LSTM', status: 'Normal', confidence: 99, latency: 38 },
];

function simulatedLogBlock() {
  return {
    total_logs: 5 + Math.floor(Math.random() * 96),
    unique_components: 1 + Math.floor(Math.random() * 8),
  };
}

function mapMetrics(payload) {
  const rows = payload?.models || payload?.comparison || payload?.metrics;
  if (!Array.isArray(rows)) return null;
  return rows.map((row, index) => ({
    name: row.name || row.model || (index === 0 ? 'Random Forest' : 'CNN-LSTM'),
    accuracy: Number(row.accuracy ?? row.Accuracy ?? 0),
    precision: Number(row.precision ?? row.Precision ?? 0),
    recall: Number(row.recall ?? row.Recall ?? 0),
    f1: Number(row.f1 ?? row.f1_score ?? row['F1-Score'] ?? 0),
    confusion: row.confusion_matrix || row.confusionMatrix || null,
  }));
}

export function useMonitor() {
  const [model, setModel] = useState('random_forest');
  const [running, setRunning] = useState(false);
  const [connection, setConnection] = useState('connecting');
  const [prediction, setPrediction] = useState({ status: 'Normal', confidence: 97, latency: 34 });
  const [points, setPoints] = useState(seed);
  const [logs, setLogs] = useState(initialLogs);
  const [total, setTotal] = useState(0);
  const [preview, setPreview] = useState(true);
  const [acknowledgedAlerts, setAcknowledgedAlerts] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState('');
  const [waking, setWaking] = useState(false);
  const busy = useRef(false);

  const addPrediction = useCallback((next) => {
    setPrediction(next);
    setTotal((count) => (preview ? 1 : count + 1));
    const date = new Date(next.timestamp);
    const health = next.status === 'Anomaly' ? 100 - next.confidence : 89 + Math.round(next.confidence / 10);
    setPoints((current) => [...(preview ? [] : current).slice(-29), {
      time: date.toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' }),
      health: Math.max(12, Math.min(100, health)), cpu: 100 - next.confidence,
    }]);
    setLogs((current) => [{
      id: `${Date.now()}-${Math.random()}`, timestamp: next.timestamp, model: next.model === 'cnn_lstm' ? 'CNN-LSTM' : 'Random Forest',
      status: next.status, confidence: next.confidence, latency: next.latency,
    }, ...(preview ? [] : current)].slice(0, 100));
    if (preview) {
      setPreview(false);
      setAcknowledgedAlerts([]);
    }
  }, [preview]);

  const runInference = useCallback(async (features = {}) => {
    if (busy.current) return null;
    busy.current = true;
    setError('');
    try {
      const next = await fetchPrediction(features, model);
      setConnection('online'); setWaking(false); addPrediction(next);
      return next;
    } catch (reason) {
      const message = reason?.name === 'AbortError' ? 'The server is taking longer than expected.' : (reason?.message || 'Unable to reach the prediction API.');
      setConnection('offline'); setWaking(true); setError(message);
      return null;
    } finally { busy.current = false; }
  }, [addPrediction, model]);

  useEffect(() => {
    let mounted = true;
    fetchHealth().then(() => { if (mounted) { setConnection('online'); setWaking(false); } })
      .catch(() => { if (mounted) { setConnection('offline'); setWaking(true); } });
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (!running) return undefined;
    // Real-time polling requests one new inference on the configured backend every 2 seconds.
    const timer = setInterval(() => { runInference(simulatedLogBlock()); }, 2000);
    return () => clearInterval(timer);
  }, [running, runInference]);

  const loadMetrics = useCallback(async () => {
    try {
      const { data } = await fetchMetrics();
      setMetrics(mapMetrics(data));
    } catch { setMetrics(null); }
  }, []);

  const acknowledgeAlert = useCallback((id) => {
    setAcknowledgedAlerts((current) => current.includes(id) ? current : [...current, id]);
  }, []);
  const reopenAlert = useCallback((id) => {
    setAcknowledgedAlerts((current) => current.filter((alertId) => alertId !== id));
  }, []);

  return {
    model, setModel, running, setRunning, connection, prediction, points, logs, total, metrics, error, waking,
    preview, acknowledgedAlerts, acknowledgeAlert, reopenAlert, runInference, loadMetrics,
  };
}
