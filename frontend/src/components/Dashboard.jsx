import { AlertTriangle, ArrowDownRight, ArrowUpRight, CheckCircle2, Clock3, Gauge, Radio, Server, ShieldCheck, Wifi, WifiOff } from 'lucide-react';
import AlertCenter from './AlertCenter.jsx';
import HealthChart from './HealthChart.jsx';
import InferencePanel from './InferencePanel.jsx';

function MetricCard({ label, value, unit, Icon, tone, note, change }) {
  const color = { violet: 'text-violet-300 bg-violet-400/10', mint: 'text-emerald-300 bg-emerald-400/10', blue: 'text-sky-300 bg-sky-400/10', amber: 'text-amber-300 bg-amber-400/10' }[tone];
  return <article className="min-w-0 rounded-2xl border border-white/[.07] bg-[#11182a] p-4 shadow-card sm:p-5">
    <div className="flex items-center justify-between"><span className="text-[11px] font-medium text-slate-400">{label}</span><span className={`grid h-8 w-8 place-items-center rounded-lg ${color}`}><Icon size={16} /></span></div>
    <div className="mt-3 flex items-baseline gap-1"><span className="font-display text-[27px] font-semibold tracking-tight text-slate-100">{value}</span>{unit && <span className="text-xs text-slate-500">{unit}</span>}</div>
    <div className="mt-1 flex items-center gap-1 text-[10px] text-slate-500">{change && (change === 'up' ? <ArrowUpRight size={12} className="text-emerald-400" /> : <ArrowDownRight size={12} className="text-rose-400" />)}{note}</div>
  </article>;
}

function ConfidenceCard({ confidence }) {
  const circumference = 2 * Math.PI * 17;
  return <article className="min-w-0 rounded-2xl border border-white/[.07] bg-[#11182a] p-4 shadow-card sm:p-5"><div className="flex items-center justify-between"><span className="text-[11px] font-medium text-slate-400">Model confidence</span><span className="grid h-8 w-8 place-items-center rounded-lg bg-violet-400/10 text-violet-300"><ShieldCheck size={16} /></span></div><div className="mt-3 flex items-center gap-3"><div className="relative grid h-12 w-12 place-items-center"><svg viewBox="0 0 42 42" className="absolute inset-0 h-full w-full -rotate-90"><circle cx="21" cy="21" r="17" fill="none" stroke="rgba(148,163,184,.15)" strokeWidth="4"/><circle cx="21" cy="21" r="17" fill="none" stroke="#9b8cff" strokeWidth="4" strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={circumference * (1 - confidence / 100)} className="transition-all duration-700"/></svg><span className="text-[10px] font-bold text-slate-100">{confidence}%</span></div><div><div className="font-display text-2xl font-semibold text-slate-100">{confidence >= 90 ? 'High' : confidence >= 70 ? 'Moderate' : 'Low'}</div><div className="mt-0.5 text-[10px] text-slate-500">Prediction certainty</div></div></div></article>;
}

export default function Dashboard({ monitor, mode, setRunning, notify }) {
  const { prediction, connection, total, points, running, model, runInference, waking, error, preview } = monitor;
  const anomaly = prediction.status === 'Anomaly';
  const modelName = model === 'cnn_lstm' ? 'CNN-LSTM' : 'Random Forest';
  return <div id="overview" className="animate-fade-in space-y-5">
    {waking && <div role="status" className="flex items-start gap-3 rounded-xl border border-amber-400/20 bg-amber-400/[.07] px-4 py-3"><span className="mt-0.5 text-amber-300"><Server size={17} /></span><div className="flex-1"><div className="text-xs font-semibold text-amber-100">Waking up the server…</div><div className="mt-0.5 text-[11px] text-amber-100/60">The API may be starting from a cold sleep. Your dashboard will reconnect automatically when it’s ready.</div>{error && <div className="mt-1 text-[10px] text-amber-100/45">{error}</div>}</div><span className="mt-1 h-2 w-2 animate-pulse rounded-full bg-amber-300" /></div>}
    {preview && <div role="status" className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-400/15 bg-sky-400/[.045] px-4 py-3"><div className="flex items-center gap-2 text-[11px] text-sky-100/80"><span className="h-1.5 w-1.5 rounded-full bg-sky-300" /><span className="font-semibold">Preview data</span><span className="text-sky-100/55">Sample telemetry and alerts are shown until the first live prediction.</span></div><span className="text-[10px] text-sky-100/45">Not production telemetry</span></div>}
    <section className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <article className="rounded-2xl border border-white/[.07] bg-[#11182a] p-4 shadow-card sm:p-5"><div className="flex items-center justify-between"><span className="text-[11px] font-medium text-slate-400">Current status</span><span className={`grid h-8 w-8 place-items-center rounded-lg ${anomaly ? 'bg-rose-400/10 text-rose-300' : 'bg-emerald-400/10 text-emerald-300'}`}>{anomaly ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}</span></div><div className="mt-3"><span className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ${anomaly ? 'fault-pulse bg-rose-400/10 text-rose-200' : 'bg-emerald-400/[.09] text-emerald-200'}`}><span className={`h-1.5 w-1.5 rounded-full ${anomaly ? 'animate-pulse bg-rose-400' : 'bg-emerald-400'}`} />{anomaly ? 'Anomaly detected' : 'All systems normal'}</span></div><div className="mt-2 text-[10px] text-slate-500">Latest {modelName} inference</div></article>
      <ConfidenceCard confidence={prediction.confidence} />
      <MetricCard label="API latency" value={prediction.latency} unit="ms" Icon={Clock3} tone="blue" note={prediction.latency < 100 ? 'Within expected range' : 'Elevated response time'} change={prediction.latency < 100 ? 'up' : 'down'} />
      <MetricCard label="Session inferences" value={total.toLocaleString()} Icon={Gauge} tone="amber" note={preview ? 'No live inferences yet' : 'Since monitoring session began'} />
    </section>
    <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.55fr)_minmax(340px,1fr)]"><HealthChart points={points} preview={preview} /><InferencePanel mode={mode} runInference={runInference} running={running} setRunning={setRunning} notify={notify} /></div>
    <AlertCenter logs={monitor.logs} acknowledgedAlerts={monitor.acknowledgedAlerts} acknowledgeAlert={monitor.acknowledgeAlert} reopenAlert={monitor.reopenAlert} />
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-white/[.06] bg-white/[.02] px-4 py-3 text-[10px] text-slate-500"><span className="flex items-center gap-2"><span className={`h-1.5 w-1.5 rounded-full ${connection === 'online' ? 'bg-emerald-400' : connection === 'offline' ? 'bg-rose-400' : 'animate-pulse bg-amber-300'}`} />{connection === 'online' ? 'API connected' : connection === 'offline' ? 'API unavailable · showing last known state' : 'Checking API connection'}</span><span className="flex items-center gap-1.5">{connection === 'online' ? <Wifi size={12} /> : <WifiOff size={12} />} Model: {modelName} · {running ? 'Polling every 2s' : mode === 'stream' ? 'Stream paused' : 'On demand'}</span></div>
  </div>;
}
