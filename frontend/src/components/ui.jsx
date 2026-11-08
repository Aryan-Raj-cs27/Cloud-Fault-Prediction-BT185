import { AlertOctagon, AlertTriangle, CheckCircle2, CircleDashed, Info } from 'lucide-react';

export function Panel({ title, subtitle, actions, children, className = '', bodyClass = 'p-4', footer }) {
  return <section className={`min-w-0 rounded border border-line bg-surface ${className}`}>
    {(title || actions) && <header className="flex flex-wrap items-start justify-between gap-2 border-b border-line px-4 py-3">
      <div className="min-w-0"><h2 className="text-[13px] font-semibold text-ink">{title}</h2>{subtitle && <p className="mt-0.5 text-2xs text-ink3">{subtitle}</p>}</div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>}
    <div className={bodyClass}>{children}</div>
    {footer && <footer className="border-t border-line px-4 py-2 text-2xs text-ink3">{footer}</footer>}
  </section>;
}

export function Stat({ label, value, unit, detail, tone }) {
  const toneCls = { ok: 'text-ok', warn: 'text-warn', crit: 'text-crit' }[tone] ?? 'text-ink';
  return <div className="rounded border border-line bg-surface px-4 py-3">
    <div className="text-xs text-ink3">{label}</div>
    <div className="mt-1 flex items-baseline gap-1"><span className={`num text-[22px] font-semibold leading-tight ${toneCls}`}>{value}</span>{unit && <span className="text-xs text-ink3">{unit}</span>}</div>
    {detail && <div className="mt-0.5 text-2xs text-ink3">{detail}</div>}
  </div>;
}

const STATUS = {
  ok: { Icon: CheckCircle2, cls: 'text-ok bg-ok/10 border-ok/25' },
  warn: { Icon: AlertTriangle, cls: 'text-warn bg-warn/10 border-warn/25' },
  crit: { Icon: AlertOctagon, cls: 'text-crit bg-crit/10 border-crit/25' },
  neutral: { Icon: CircleDashed, cls: 'text-ink2 bg-sunken border-line' },
};

/** Status pill: always icon + label, never colour alone. */
export function Badge({ status = 'neutral', children }) {
  const { Icon, cls } = STATUS[status];
  return <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-0.5 text-2xs font-medium ${cls}`}><Icon size={11} aria-hidden />{children}</span>;
}

export const riskStatus = (level) => ({ high: 'crit', elevated: 'warn', low: 'ok' }[level] ?? 'neutral');

export function Segmented({ options, value, onChange, label }) {
  return <div role="radiogroup" aria-label={label} className="inline-flex rounded border border-line bg-sunken p-0.5">
    {options.map((o) => <button key={o.value} role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
      className={`rounded-[4px] px-2.5 py-1 text-xs ${value === o.value ? 'bg-surface font-medium text-ink shadow-sm' : 'text-ink2 hover:text-ink'}`}>{o.label}</button>)}
  </div>;
}

export function Button({ variant = 'secondary', className = '', children, ...props }) {
  const cls = variant === 'primary'
    ? 'border-accent bg-accent text-white hover:bg-accent/90'
    : 'border-line bg-surface text-ink hover:bg-sunken';
  return <button {...props} className={`inline-flex items-center gap-1.5 rounded border px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${cls} ${className}`}>{children}</button>;
}

export function Empty({ title, children }) {
  return <div className="flex items-start gap-2 rounded border border-dashed border-line bg-sunken px-4 py-6 text-xs text-ink2"><Info size={14} className="mt-0.5 shrink-0 text-ink3" /><div><div className="font-medium text-ink">{title}</div>{children && <div className="mt-1 text-ink3">{children}</div>}</div></div>;
}

export function Table({ head, children, minWidth = 560 }) {
  return <div className="overflow-x-auto"><table className="w-full border-collapse text-left text-xs" style={{ minWidth }}>
    <thead><tr className="border-b border-line">{head.map((h, i) => <th key={i} className={`whitespace-nowrap px-3 py-2 text-2xs font-medium uppercase tracking-wide text-ink3 ${h.right ? 'text-right' : ''}`}>{h.label ?? h}</th>)}</tr></thead>
    <tbody className="divide-y divide-line">{children}</tbody>
  </table></div>;
}

export const Td = ({ right, mono, className = '', children, ...rest }) => <td {...rest} className={`px-3 py-2 align-middle ${right ? 'num whitespace-nowrap text-right' : ''} ${mono ? 'font-mono text-[11.5px]' : ''} ${className}`}>{children}</td>;

export function ChartTooltip({ active, payload, label, format }) {
  if (!active || !payload?.length) return null;
  return <div className="rounded border border-line bg-surface px-2.5 py-1.5 text-2xs shadow-md">
    {label != null && <div className="mb-0.5 text-ink3">{label}</div>}
    {payload.map((p) => <div key={p.dataKey} className="num text-ink">{format ? format(p) : `${p.name}: ${p.value}`}</div>)}
  </div>;
}

export const pct = (v, digits = 1) => (v == null || Number.isNaN(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(digits)}%`);
export const fmt = (v, digits = 0) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }));
/** Parse "mean ± std" strings produced by the evaluation scripts. */
export const meanStd = (text) => { const [m, s] = String(text ?? '').split('±').map(Number); return { mean: m, std: Number.isFinite(s) ? s : 0 }; };
