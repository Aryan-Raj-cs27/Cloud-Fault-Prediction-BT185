import { useState } from 'react';
import { Activity, Bell, CheckCircle2, Menu, Moon, Sun, X } from 'lucide-react';
import Dashboard from './components/Dashboard.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import Performance from './components/Performance.jsx';
import PredictionLogs from './components/PredictionLogs.jsx';
import Sidebar from './components/Sidebar.jsx';
import { useMonitor } from './hooks/useMonitor.js';

const navTabs = [{ id: 'overview', label: 'Overview' }, { id: 'performance', label: 'Model analytics' }, { id: 'logs', label: 'Prediction history' }];

function Toast({ toast, dismiss }) {
  if (!toast) return null;
  return <div role="status" className={`fixed bottom-5 right-5 z-[60] flex max-w-sm items-center gap-3 rounded-xl border px-4 py-3 shadow-2xl animate-fade-in ${toast.type === 'error' ? 'border-rose-400/20 bg-[#241522] text-rose-100' : 'border-emerald-400/20 bg-[#10231f] text-emerald-100'}`}><CheckCircle2 size={17} className={toast.type === 'error' ? 'text-rose-300' : 'text-emerald-300'} /><span className="flex-1 text-xs">{toast.message}</span><button aria-label="Dismiss notification" onClick={dismiss} className="text-slate-400 hover:text-white"><X size={14} /></button></div>;
}

function App() {
  const monitor = useMonitor();
  const [mode, setMode] = useState('stream');
  const [tab, setTab] = useState('overview');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [lightMode, setLightMode] = useState(false);
  const [toast, setToast] = useState(null);
  const notify = (message, type = 'success') => {
    setToast({ message, type });
    window.clearTimeout(window.toastTimeout);
    window.toastTimeout = window.setTimeout(() => setToast(null), 4200);
  };
  const changeTab = (next) => {
    setTab(next);
    document.getElementById(next)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return <div className={`${lightMode ? 'light-theme' : ''} dashboard-shell min-h-screen bg-[#0b1020] text-slate-100`}>
    <Sidebar model={monitor.model} setModel={monitor.setModel} mode={mode} setMode={setMode} running={monitor.running} setRunning={monitor.setRunning} mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
    <div className="min-h-screen lg:pl-[264px]">
      <header className="sticky top-0 z-20 border-b border-white/[.06] bg-[#0b1020]/90 px-4 backdrop-blur-xl sm:px-7 lg:px-9">
        <div className="flex h-[68px] items-center gap-3"><button aria-label="Open navigation" onClick={() => setMobileOpen(true)} className="rounded-lg p-2 text-slate-400 hover:bg-white/5 lg:hidden"><Menu size={19} /></button><div className="min-w-0 flex-1"><div className="hidden text-[10px] font-medium text-slate-500 sm:block">Cloud infrastructure <span className="mx-1.5 text-slate-700">/</span> <span className="text-slate-400">Monitoring</span></div><div className="flex items-center gap-2 sm:mt-1"><span className="font-display text-sm font-semibold text-slate-100">{navTabs.find((item) => item.id === tab)?.label}</span><span className="hidden rounded-md bg-white/[.04] px-1.5 py-0.5 text-[9px] text-slate-500 sm:inline">LIVE</span></div></div><div className="hidden items-center gap-1 rounded-lg border border-white/[.07] bg-white/[.025] p-1 md:flex">{navTabs.map((item) => <button key={item.id} onClick={() => changeTab(item.id)} className={`rounded-md px-3 py-1.5 text-[10px] font-medium transition ${tab === item.id ? 'bg-violet-500/15 text-violet-200' : 'text-slate-500 hover:text-slate-200'}`}>{item.label}</button>)}</div><div className="flex items-center gap-1.5"><span className="mr-1 hidden items-center gap-1.5 rounded-full border border-emerald-400/10 bg-emerald-400/[.05] px-2.5 py-1.5 text-[9px] font-medium text-emerald-300 sm:flex"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />LIVE</span><button aria-label="Toggle color theme" onClick={() => setLightMode(!lightMode)} className="rounded-lg p-2 text-slate-400 transition hover:bg-white/[.06] hover:text-slate-200">{lightMode ? <Sun size={17} /> : <Moon size={16} />}</button><button aria-label="Notifications" onClick={() => notify(monitor.waking ? 'Backend is waking up; retrying automatically.' : 'You’re all caught up.', monitor.waking ? 'error' : 'success')} className="relative rounded-lg p-2 text-slate-400 transition hover:bg-white/[.06] hover:text-slate-200"><Bell size={16} />{monitor.waking && <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-amber-400" />}</button><div className="ml-1 grid h-8 w-8 place-items-center rounded-full border border-white/10 bg-gradient-to-br from-indigo-400 to-violet-600 text-[10px] font-bold text-white">CF</div></div></div>
        <div className="-mx-1 flex gap-1 overflow-x-auto pb-2 md:hidden">{navTabs.map((item) => <button key={item.id} onClick={() => changeTab(item.id)} className={`shrink-0 rounded-lg px-3 py-1.5 text-[10px] font-medium ${tab === item.id ? 'bg-violet-500/15 text-violet-200' : 'text-slate-500'}`}>{item.label}</button>)}</div>
      </header>
      <main className="mx-auto max-w-[1600px] space-y-6 px-4 py-6 sm:px-7 sm:py-7 lg:px-9 lg:py-8">
        <div className="flex flex-wrap items-end justify-between gap-3"><div><div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[.19em] text-violet-300"><Activity size={13} /> Infrastructure intelligence</div><h1 className="font-display mt-2 text-2xl font-semibold tracking-tight text-slate-50 sm:text-[30px]">Cloud health overview</h1><p className="mt-1.5 text-xs text-slate-500">Monitor telemetry, detect faults, and evaluate model performance.</p></div><div className="flex items-center gap-2 rounded-lg border border-white/[.07] bg-white/[.025] px-3 py-2 text-[10px] text-slate-400"><span className={`h-1.5 w-1.5 rounded-full ${monitor.connection === 'online' ? 'bg-emerald-400' : 'bg-amber-400'}`} />Updated {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div></div>
        {tab === 'overview' && <Dashboard monitor={monitor} mode={mode} setRunning={monitor.setRunning} notify={notify} />}
        {tab === 'performance' && <Performance metrics={monitor.metrics} loadMetrics={monitor.loadMetrics} />}
        {tab === 'logs' && <PredictionLogs logs={monitor.logs} preview={monitor.preview} />}
        {tab === 'overview' && <><div className="flex flex-wrap items-center justify-between gap-2 pt-1"><h2 className="font-display text-sm font-semibold text-slate-200">Recent predictions</h2><button onClick={() => changeTab('logs')} className="text-[10px] font-semibold text-violet-300 hover:text-violet-200">View all history →</button></div><PredictionLogs logs={monitor.logs.slice(0, 5)} preview={monitor.preview} /></>}
        <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[.05] pt-4 text-[9px] text-slate-600"><span>Sentinel · IoT server health monitoring</span><span>Machine learning powered fault detection</span></footer>
      </main>
    </div>
    <Toast toast={toast} dismiss={() => setToast(null)} />
  </div>;
}

export default function AppRoot() {
  return <ErrorBoundary><App /></ErrorBoundary>;
}
