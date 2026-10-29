import { Activity, BrainCircuit, ChevronDown, CircleHelp, Cpu, Database, Gauge, Radio, Settings2, UploadCloud, Waves, X } from 'lucide-react';

const modes = [
  { id: 'stream', label: 'Live stream', detail: 'Real-time inference', Icon: Radio },
  { id: 'manual', label: 'Single sample', detail: 'Enter telemetry', Icon: Settings2 },
  { id: 'batch', label: 'Batch upload', detail: 'Evaluate a CSV', Icon: UploadCloud },
];

export default function Sidebar({ model, setModel, mode, setMode, running, setRunning, mobileOpen, onClose }) {
  return <>
    {mobileOpen && <button aria-label="Close navigation" onClick={onClose} className="fixed inset-0 z-30 bg-black/60 lg:hidden" />}
    <aside className={`fixed inset-y-0 left-0 z-40 flex w-[264px] flex-col border-r border-white/[.07] bg-[#0c1222] px-4 py-5 transition-transform duration-300 lg:translate-x-0 ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}`}>
      <div className="flex items-center justify-between px-2">
        <a href="#overview" className="flex items-center gap-3" onClick={onClose}>
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 shadow-glow"><Activity size={21} className="text-white" /></span>
          <span><span className="font-display block text-[17px] font-bold tracking-tight text-white">sentinel<span className="text-violet-400">.</span></span><span className="block text-[10px] font-semibold uppercase tracking-[.19em] text-slate-500">Cloud observability</span></span>
        </a>
        <button onClick={onClose} className="rounded-lg p-2 text-slate-400 hover:bg-white/5 lg:hidden" aria-label="Close sidebar"><X size={18} /></button>
      </div>
      <div className="mt-9 px-2 text-[10px] font-bold uppercase tracking-[.2em] text-slate-500">Workspace</div>
      <button className="mt-3 flex w-full items-center gap-3 rounded-xl border border-white/[.08] bg-white/[.03] p-3 text-left hover:bg-white/[.05]">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-cyan-400/10 text-cyan-300"><Database size={16} /></span>
        <span className="min-w-0 flex-1"><span className="block truncate text-xs font-semibold text-slate-200">Production cluster</span><span className="mt-1 flex items-center gap-1.5 text-[10px] text-slate-500"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> us-east-1 · 8 nodes</span></span>
        <ChevronDown size={14} className="text-slate-500" />
      </button>
      <div className="mt-8 px-2 text-[10px] font-bold uppercase tracking-[.2em] text-slate-500">Monitoring</div>
      <a href="#overview" onClick={onClose} className="mt-3 flex items-center gap-3 rounded-xl bg-violet-500/10 px-3 py-2.5 text-sm font-semibold text-violet-200"><Gauge size={17} /> Overview <span className="ml-auto h-1.5 w-1.5 rounded-full bg-violet-400" /></a>
      <a href="#performance" onClick={onClose} className="mt-1 flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-slate-400 transition hover:bg-white/[.04] hover:text-slate-200"><BrainCircuit size={17} /> Model analytics</a>
      <a href="#logs" onClick={onClose} className="mt-1 flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-slate-400 transition hover:bg-white/[.04] hover:text-slate-200"><Waves size={17} /> Prediction history</a>
      <div className="mt-8 px-2 text-[10px] font-bold uppercase tracking-[.2em] text-slate-500">Inference setup</div>
      <label className="mt-3 block px-1 text-[11px] font-medium text-slate-400" htmlFor="model-select">Active model</label>
      <div className="relative mt-2">
        <BrainCircuit size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-violet-300" />
        <select id="model-select" value={model} onChange={(event) => setModel(event.target.value)} className="w-full appearance-none rounded-xl border border-white/[.09] bg-[#131b2e] py-3 pl-10 pr-9 text-xs font-semibold text-slate-200 outline-none transition focus:border-violet-400/50">
          <option value="random_forest">Random Forest (Baseline)</option>
          <option value="cnn_lstm">CNN-LSTM (Deep Learning)</option>
        </select><ChevronDown size={14} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500" />
      </div>
      <div className="mt-7 px-1 text-[11px] font-medium text-slate-400">Inference mode</div>
      <div className="mt-2 space-y-1">
        {modes.map(({ id, label, detail, Icon }) => <button key={id} onClick={() => { setMode(id); onClose?.(); }} className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition ${mode === id ? 'border-violet-400/25 bg-violet-500/[.08]' : 'border-transparent hover:bg-white/[.035]'}`}>
          <span className={`grid h-8 w-8 place-items-center rounded-lg ${mode === id ? 'bg-violet-500/15 text-violet-300' : 'bg-white/[.04] text-slate-500'}`}><Icon size={15} /></span>
          <span className="min-w-0 flex-1"><span className={`block text-xs font-semibold ${mode === id ? 'text-slate-100' : 'text-slate-400'}`}>{label}</span><span className="mt-0.5 block text-[10px] text-slate-600">{detail}</span></span>
          {id === 'stream' && <span className={`h-1.5 w-1.5 rounded-full ${running ? 'animate-pulse bg-emerald-400' : 'bg-slate-600'}`} />}
        </button>)}
      </div>
      <div className="mt-4 rounded-xl border border-white/[.07] bg-white/[.025] p-3">
        <div className="flex items-center justify-between"><span className="text-[11px] font-semibold text-slate-300">Stream polling</span><button aria-label={running ? 'Pause live stream' : 'Start live stream'} onClick={() => setRunning(!running)} className={`relative h-[21px] w-[38px] rounded-full transition ${running ? 'bg-emerald-500' : 'bg-slate-700'}`}><span className={`absolute top-[3px] h-[15px] w-[15px] rounded-full bg-white transition-all ${running ? 'left-[20px]' : 'left-[3px]'}`} /></button></div>
        <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-slate-500"><Cpu size={12} /> Polls every 2 seconds</div>
      </div>
      <div className="mt-auto">
        <div className="rounded-xl border border-white/[.07] bg-gradient-to-br from-violet-500/[.11] to-transparent p-3.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-200"><CircleHelp size={15} className="text-violet-300" /> Need a hand?</div>
          <p className="mb-0 mt-1.5 text-[10px] leading-relaxed text-slate-500">Check the API connection and integration guide in the project README.</p>
        </div>
        <div className="mt-4 flex items-center gap-2.5 border-t border-white/[.07] px-1 pt-4"><div className="grid h-8 w-8 place-items-center rounded-full bg-gradient-to-br from-orange-300 to-rose-400 text-[11px] font-bold text-white">PK</div><div className="flex-1"><div className="text-xs font-semibold text-slate-200">Project workspace</div><div className="text-[10px] text-slate-500">IoT monitoring team</div></div><button aria-label="Settings" className="rounded-lg p-1.5 text-slate-500 hover:bg-white/5"><Settings2 size={15} /></button></div>
      </div>
    </aside>
  </>;
}
