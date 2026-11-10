import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchHealth, fetchMetrics, fetchPrediction, fetchSample, fetchSimulation } from '../lib/api.js';

/** Console state: live replay of real held-out HDFS blocks + evaluation/simulation results. */
export function useMonitor() {
  const [obsWindow, setObsWindow] = useState('w60');
  const [running, setRunning] = useState(false);
  const [connection, setConnection] = useState('connecting');
  const [health, setHealth] = useState(null);
  const [logs, setLogs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [error, setError] = useState('');
  const busy = useRef(false);

  const predictLines = useCallback(async (lines, truth = null, blockId = null) => {
    if (busy.current) return null;
    busy.current = true;
    setError('');
    try {
      const next = { ...(await fetchPrediction(lines, obsWindow, truth, blockId)), lines };
      setConnection('online');
      setLogs((current) => [next, ...current].slice(0, 300));
      setSelectedId(next.id);
      return next;
    } catch (reason) {
      setError(reason?.message || 'Unable to reach the API.');
      setConnection((c) => (reason?.status === 0 ? 'offline' : c));
      return null;
    } finally { busy.current = false; }
  }, [obsWindow]);

  const loadSample = useCallback(async (label) => (await fetchSample(label)).data, []);

  /** Score one real held-out block; its ground truth is kept for display only. */
  const replayOne = useCallback(async (label) => {
    try {
      const sample = await loadSample(label);
      return predictLines(sample.log_lines, sample.label, sample.block_id);
    } catch (reason) {
      setError(reason?.message || 'Could not load a sample block.');
      return null;
    }
  }, [loadSample, predictLines]);

  const loadMetrics = useCallback(async (tag) => {
    try { setMetrics((await fetchMetrics(tag)).data); } catch { setMetrics(null); }
  }, []);
  const loadSimulation = useCallback(async () => {
    try { setSimulation((await fetchSimulation()).data); } catch { setSimulation(null); }
  }, []);

  useEffect(() => {
    fetchHealth()
      .then(({ data }) => { setHealth(data); setConnection(data.status === 'ok' ? 'online' : 'degraded'); })
      .catch(() => setConnection('offline'));
    loadMetrics();
    loadSimulation();
  }, [loadMetrics, loadSimulation]);

  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(() => { replayOne(); }, 2000);
    return () => clearInterval(timer);
  }, [running, replayOne]);

  const labelled = logs.filter((l) => l.truth);
  const session = {
    total: logs.length,
    labelled: labelled.length,
    tp: labelled.filter((l) => l.truth === 'Anomaly' && l.status === 'Anomaly').length,
    fn: labelled.filter((l) => l.truth === 'Anomaly' && l.status !== 'Anomaly').length,
    fp: labelled.filter((l) => l.truth === 'Normal' && l.status === 'Anomaly').length,
    tn: labelled.filter((l) => l.truth === 'Normal' && l.status !== 'Anomaly').length,
  };

  return {
    obsWindow, setObsWindow, running, setRunning, connection, health, logs, session, error,
    selected: logs.find((l) => l.id === selectedId) ?? logs[0] ?? null, setSelectedId,
    metrics, simulation, predictLines, replayOne, loadSample, loadMetrics, loadSimulation,
  };
}
