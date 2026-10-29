import { useMemo, useState } from 'react';
import { BellRing, Check, CircleCheck, Clock3, RotateCcw } from 'lucide-react';

function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export default function AlertCenter({ logs, acknowledgedAlerts, acknowledgeAlert, reopenAlert }) {
  const [filter, setFilter] = useState('active');
  const alerts = useMemo(() => logs.filter((log) => log.status === 'Anomaly'), [logs]);
  const openCount = alerts.filter((alert) => !acknowledgedAlerts.includes(alert.id)).length;
  const visibleAlerts = alerts.filter((alert) => (
    filter === 'active' ? !acknowledgedAlerts.includes(alert.id) : acknowledgedAlerts.includes(alert.id)
  ));

  return <section id="alerts" className="rounded-2xl border border-white/[.07] bg-[#11182a] shadow-card">
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[.07] px-5 py-5 sm:px-6">
      <div className="flex items-start gap-3">
        <span className={`grid h-9 w-9 place-items-center rounded-xl ${openCount ? 'bg-rose-400/10 text-rose-300' : 'bg-emerald-400/10 text-emerald-300'}`}>
          {openCount ? <BellRing size={17} /> : <CircleCheck size={17} />}
        </span>
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-semibold text-slate-100">Alert center</h2>
            <span className={`rounded-full px-2 py-0.5 text-[9px] font-semibold ${openCount ? 'bg-rose-400/10 text-rose-300' : 'bg-emerald-400/[.08] text-emerald-300'}`}>
              {openCount} open
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-500">Anomaly detections from this monitoring session</p>
        </div>
      </div>
      <div className="flex rounded-lg border border-white/[.07] bg-white/[.025] p-1" aria-label="Filter alerts">
        {[
          { id: 'active', label: `Active (${openCount})` },
          { id: 'acknowledged', label: `Acknowledged (${alerts.length - openCount})` },
        ].map((item) => (
          <button key={item.id} onClick={() => setFilter(item.id)} aria-pressed={filter === item.id}
            className={`rounded-md px-2.5 py-1.5 text-[10px] font-medium transition ${filter === item.id ? 'bg-violet-500/15 text-violet-200' : 'text-slate-500 hover:text-slate-200'}`}>
            {item.label}
          </button>
        ))}
      </div>
    </div>
    {visibleAlerts.length ? <div className="divide-y divide-white/[.045]">
      {visibleAlerts.slice(0, 6).map((alert) => {
        const acknowledged = acknowledgedAlerts.includes(alert.id);
        return <article key={alert.id} className="flex flex-wrap items-center gap-3 px-5 py-4 sm:px-6">
          <span className={`h-2 w-2 shrink-0 rounded-full ${acknowledged ? 'bg-slate-500' : 'animate-pulse bg-rose-400'}`} />
          <div className="min-w-[180px] flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-semibold text-slate-200">Anomaly detected</span>
              <span className="rounded-md bg-rose-400/[.08] px-1.5 py-0.5 text-[9px] font-medium text-rose-300">Model alert</span>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-slate-500">
              <span>{alert.model}</span>
              <span className="flex items-center gap-1"><Clock3 size={11} />{formatTime(alert.timestamp)}</span>
              <span>{alert.confidence}% confidence · {alert.latency} ms</span>
            </div>
          </div>
          {acknowledged
            ? <button onClick={() => reopenAlert(alert.id)} className="inline-flex items-center gap-1.5 rounded-lg border border-white/[.08] px-2.5 py-2 text-[10px] font-semibold text-slate-400 transition hover:bg-white/[.05] hover:text-slate-200"><RotateCcw size={12} />Reopen</button>
            : <button onClick={() => acknowledgeAlert(alert.id)} className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-400/15 bg-emerald-400/[.05] px-2.5 py-2 text-[10px] font-semibold text-emerald-300 transition hover:bg-emerald-400/10"><Check size={12} />Acknowledge</button>}
        </article>;
      })}
      {visibleAlerts.length > 6 && <div className="px-5 py-3 text-center text-[10px] text-slate-500 sm:px-6">Showing 6 of {visibleAlerts.length} alerts · view prediction history for all events</div>}
    </div> : <div className="flex items-center gap-3 px-5 py-8 sm:px-6">
      <CircleCheck size={17} className="text-emerald-300" />
      <div><div className="text-xs font-semibold text-slate-300">{filter === 'active' ? 'No open alerts' : 'No acknowledged alerts'}</div><div className="mt-1 text-[10px] text-slate-500">{filter === 'active' ? 'Anomaly detections will appear here when the model reports them.' : 'Acknowledged detections will appear here.'}</div></div>
    </div>}
    <div className="flex items-center gap-2 border-t border-white/[.05] px-5 py-3 text-[10px] text-slate-600 sm:px-6">
      <span className="h-1.5 w-1.5 rounded-full bg-violet-400" /> Alert actions are tracked for this session
    </div>
  </section>;
}
