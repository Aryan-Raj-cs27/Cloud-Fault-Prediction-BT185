import { useRef, useState } from 'react';
import { Activity, Check, FileSpreadsheet, Play, UploadCloud } from 'lucide-react';

const initial = { cpu_load: 42, memory_usage: 68, network_io: 31, disk_usage: 54, temperature: 62 };
const fields = [
  { key: 'cpu_load', label: 'CPU load', unit: '%', min: 0, max: 100 },
  { key: 'memory_usage', label: 'Memory usage', unit: '%', min: 0, max: 100 },
  { key: 'network_io', label: 'Network I/O', unit: 'MB/s', min: 0, max: 100 },
  { key: 'disk_usage', label: 'Disk usage', unit: '%', min: 0, max: 100 },
  { key: 'temperature', label: 'Temperature', unit: '°C', min: 20, max: 100 },
];

function parseCsv(text) {
  const rows = text.trim().split(/\r?\n/).map((line) => line.split(',').map((value) => value.trim()));
  if (rows.length < 2) throw new Error('Add a header row and at least one data row.');
  const headers = rows[0].map((header) => header.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, ''));
  return rows.slice(1).filter((row) => row.some(Boolean)).map((row) => Object.fromEntries(headers.map((key, index) => [key, Number(row[index]) || 0])));
}

export default function InferencePanel({ mode, runInference, running, setRunning, notify }) {
  const [values, setValues] = useState(initial);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const inputRef = useRef(null);

  const evaluateSample = async () => {
    setBusy(true);
    try {
      const result = await runInference(values);
      if (result) notify?.('Sample evaluated successfully', 'success');
      else notify?.('The backend is unavailable. The dashboard will retry on the next live poll.', 'error');
    }
    finally { setBusy(false); }
  };

  const evaluateFile = async () => {
    if (!file) return;
    setBusy(true); setBatchResult(null);
    try {
      const samples = parseCsv(await file.text());
      let faults = 0;
      for (const sample of samples) {
        const result = await runInference(sample);
        if (!result) throw new Error('The backend is unavailable. Batch evaluation stopped before all rows were processed.');
        if (result?.status === 'Anomaly') faults += 1;
      }
      setBatchResult({ total: samples.length, faults });
      notify?.(`Evaluated ${samples.length} CSV records`, 'success');
    } catch (error) { notify?.(error.message || 'Could not parse CSV.', 'error'); }
    finally { setBusy(false); }
  };

  if (mode === 'stream') return <section className="rounded-2xl border border-white/[.07] bg-[#11182a] p-5 shadow-card sm:p-6">
    <div className="flex items-start gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl bg-emerald-400/10 text-emerald-300"><Activity size={17} /></span><div className="flex-1"><h2 className="text-sm font-semibold text-slate-100">Live inference stream</h2><p className="mt-1 text-xs text-slate-500">Poll the prediction API every 2 seconds</p></div><span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${running ? 'bg-emerald-400/10 text-emerald-300' : 'bg-slate-700/50 text-slate-400'}`}>{running ? 'STREAMING' : 'PAUSED'}</span></div>
    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/[.06] bg-[#0c1222] p-4"><div><div className="text-xs font-medium text-slate-300">Automated health checks</div><div className="mt-1 text-[11px] text-slate-500">New telemetry is sent to the active model on each poll.</div></div><button onClick={() => setRunning(!running)} className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition ${running ? 'border border-white/10 bg-white/[.04] text-slate-300 hover:bg-white/[.08]' : 'bg-violet-500 text-white shadow-glow hover:bg-violet-400'}`}><Play size={14} className={running ? 'hidden' : 'fill-current'} />{running ? 'Pause stream' : 'Start stream'}</button></div>
  </section>;

  if (mode === 'manual') return <section className="rounded-2xl border border-white/[.07] bg-[#11182a] p-5 shadow-card sm:p-6">
    <div className="flex items-start gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl bg-violet-400/10 text-violet-300"><Activity size={17} /></span><div><h2 className="text-sm font-semibold text-slate-100">Evaluate a single sample</h2><p className="mt-1 text-xs text-slate-500">Adjust telemetry values, then request a prediction</p></div></div>
    <div className="mt-5 grid gap-x-6 gap-y-4 sm:grid-cols-2">
      {fields.map((field) => <label key={field.key}><span className="flex items-center justify-between text-[11px] font-medium text-slate-400"><span>{field.label}</span><span className="font-semibold text-slate-200">{values[field.key]} <span className="font-normal text-slate-500">{field.unit}</span></span></span><input aria-label={field.label} className="range-control mt-2 w-full cursor-pointer" type="range" min={field.min} max={field.max} value={values[field.key]} onChange={(event) => setValues((current) => ({ ...current, [field.key]: Number(event.target.value) }))} /></label>)}
    </div>
    <button disabled={busy} onClick={evaluateSample} className="mt-5 inline-flex items-center gap-2 rounded-lg bg-violet-500 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-violet-400 disabled:cursor-wait disabled:opacity-60"><Play size={14} className="fill-current" />{busy ? 'Evaluating…' : 'Run inference'}</button>
  </section>;

  return <section className="rounded-2xl border border-white/[.07] bg-[#11182a] p-5 shadow-card sm:p-6">
    <div className="flex items-start gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl bg-cyan-400/10 text-cyan-300"><FileSpreadsheet size={17} /></span><div><h2 className="text-sm font-semibold text-slate-100">Batch CSV evaluation</h2><p className="mt-1 text-xs text-slate-500">Upload telemetry records for sequential inference</p></div></div>
    <input ref={inputRef} type="file" accept=".csv,text/csv" className="hidden" onChange={(event) => { setFile(event.target.files?.[0] || null); setBatchResult(null); }} />
    <button onClick={() => inputRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); const picked = event.dataTransfer.files?.[0]; if (picked && picked.name.toLowerCase().endsWith('.csv')) setFile(picked); else notify?.('Please select a CSV file.', 'error'); }} className={`mt-5 flex min-h-32 w-full flex-col items-center justify-center rounded-xl border border-dashed px-4 py-5 transition ${dragging ? 'border-violet-400 bg-violet-400/10' : 'border-white/[.14] bg-[#0c1222] hover:border-violet-400/50 hover:bg-violet-400/[.03]'}`}>
      <span className="grid h-9 w-9 place-items-center rounded-xl bg-violet-500/10 text-violet-300"><UploadCloud size={18} /></span><span className="mt-2 text-xs font-semibold text-slate-200">{file?.name || 'Drop your CSV here, or browse'}</span><span className="mt-1 text-[10px] text-slate-500">CSV with a header row · one record per line</span>
    </button>
    <div className="mt-3 flex flex-wrap items-center justify-between gap-3"><span className="text-[10px] text-slate-500">Columns are passed through to the configured API.</span><button disabled={!file || busy} onClick={evaluateFile} className="inline-flex items-center gap-2 rounded-lg bg-violet-500 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-violet-400 disabled:cursor-not-allowed disabled:opacity-40"><Play size={13} className="fill-current" />{busy ? 'Evaluating…' : 'Evaluate file'}</button></div>
    {batchResult && <div className="mt-4 flex items-center gap-2 rounded-lg bg-emerald-400/[.08] p-3 text-xs text-emerald-200"><Check size={15} />{batchResult.total} records evaluated · {batchResult.faults} anomalies detected</div>}
  </section>;
}
