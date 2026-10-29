import { useEffect, useState } from 'react';
import { BarChart3, BrainCircuit, Check, ChevronDown, Crosshair, Info, Target } from 'lucide-react';

const sampleMetrics = [
  { name: 'Random Forest', accuracy: 96.8, precision: 95.1, recall: 93.7, f1: 94.4, confusion: [[412, 14], [21, 153]] },
  { name: 'CNN-LSTM', accuracy: 98.2, precision: 97.4, recall: 96.1, f1: 96.7, confusion: [[419, 7], [11, 163]] },
];
const metricInfo = { Accuracy: 'Correct predictions / all predictions', Precision: 'Correct anomaly alerts / all alerts', Recall: 'Detected anomalies / actual anomalies', 'F1-Score': 'Harmonic mean of precision and recall' };

function asPercent(value) { const number = Number(value); return `${number <= 1 ? (number * 100).toFixed(1) : number.toFixed(1)}%`; }

function ConfusionMatrix({ matrix }) {
  const values = Array.isArray(matrix) ? matrix : [[412, 14], [21, 153]];
  const flat = values.flat().map(Number); const maximum = Math.max(1, ...flat);
  return <div className="mt-4 grid grid-cols-[52px_1fr_1fr] gap-1.5 text-center text-[10px]"><div /><div className="pb-1 text-slate-500">Pred. normal</div><div className="pb-1 text-slate-500">Pred. fault</div><div className="flex items-center justify-start text-slate-500">Actual<br />normal</div>{values[0].map((value, index) => <div key={`a-${index}`} className="rounded-lg p-3 font-display text-sm font-semibold text-slate-100" style={{ backgroundColor: `rgba(124, 108, 255, ${0.08 + (Number(value) / maximum) * 0.34})` }}>{value}</div>)}<div className="flex items-center justify-start text-slate-500">Actual<br />fault</div>{(values[1] || []).map((value, index) => <div key={`b-${index}`} className="rounded-lg p-3 font-display text-sm font-semibold text-slate-100" style={{ backgroundColor: `rgba(124, 108, 255, ${0.08 + (Number(value) / maximum) * 0.34})` }}>{value}</div>)}</div>;
}

export default function Performance({ metrics: apiMetrics, loadMetrics }) {
  const [selected, setSelected] = useState(0);
  const [showInfo, setShowInfo] = useState(false);
  useEffect(() => { loadMetrics?.(); }, [loadMetrics]);
  const metrics = apiMetrics?.length ? apiMetrics : sampleMetrics;
  const model = metrics[Math.min(selected, metrics.length - 1)];
  const keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
  return <section id="performance" className="animate-fade-in space-y-5">
    <div><div className="flex items-center gap-2 text-sm font-semibold text-slate-100"><BarChart3 size={17} className="text-violet-300" /> Model performance</div><p className="mt-1 text-xs text-slate-500">Compare classification metrics and inspect prediction outcomes.</p></div>
    {!apiMetrics?.length && <div className="flex items-center gap-2 rounded-lg border border-sky-400/10 bg-sky-400/[.04] px-3 py-2 text-[10px] text-sky-200/70"><Info size={13} />Preview metrics shown until the backend returns a model comparison payload.</div>}
    <div className="grid gap-5 xl:grid-cols-[1.4fr_1fr]">
      <div className="rounded-2xl border border-white/[.07] bg-[#11182a] p-5 shadow-card sm:p-6"><div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-sm font-semibold text-slate-100">Model comparison</h2><p className="mt-1 text-xs text-slate-500">Classification quality by evaluation metric</p></div><span className="rounded-lg border border-white/[.08] bg-white/[.03] px-3 py-2 text-[10px] text-slate-400">Evaluation set <ChevronDown size={12} className="ml-2 inline" /></span></div>
        <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[460px] text-left"><thead><tr className="border-b border-white/[.07] text-[10px] uppercase tracking-wider text-slate-500"><th className="pb-3 font-medium">Model</th>{keys.map((key) => <th key={key} className="pb-3 text-right font-medium">{key}</th>)}</tr></thead><tbody>{metrics.map((item, index) => <tr key={item.name} className={`border-b border-white/[.045] last:border-0 ${selected === index ? 'bg-violet-400/[.025]' : ''}`}><td className="py-4"><button onClick={() => setSelected(index)} className="flex items-center gap-2.5 text-xs font-semibold text-slate-200"><span className={`grid h-7 w-7 place-items-center rounded-lg ${index === 0 ? 'bg-amber-400/10 text-amber-300' : 'bg-violet-400/10 text-violet-300'}`}><BrainCircuit size={14} /></span>{item.name}{selected === index && <Check size={12} className="text-violet-300" />}</button></td>{[item.accuracy, item.precision, item.recall, item.f1].map((value, metricIndex) => <td key={metricIndex} className="py-4 text-right font-display text-xs font-medium text-slate-300">{asPercent(value)}</td>)}</tr>)}</tbody></table></div>
        <div className="mt-4 grid gap-2 sm:grid-cols-2">{keys.map((key, index) => { const val = [model.accuracy, model.precision, model.recall, model.f1][index]; return <div key={key} className="rounded-xl bg-[#0c1222] p-3"><div className="flex justify-between text-[10px] text-slate-500"><span>{key}</span><span className="font-semibold text-slate-200">{asPercent(val)}</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/[.06]"><div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-indigo-400 transition-all duration-700" style={{ width: `${Math.min(100, Number(val) <= 1 ? Number(val) * 100 : Number(val))}%` }} /></div></div>; })}</div>
      </div>
      <div className="rounded-2xl border border-white/[.07] bg-[#11182a] p-5 shadow-card sm:p-6"><div className="flex items-start justify-between"><div><div className="flex items-center gap-2 text-sm font-semibold text-slate-100"><Crosshair size={16} className="text-cyan-300" /> Confusion matrix</div><p className="mt-1 text-xs text-slate-500">{model.name} · actual vs predicted</p></div><button onClick={() => setShowInfo(!showInfo)} aria-label="Explain confusion matrix" className="rounded-lg p-1.5 text-slate-500 hover:bg-white/5"><Info size={15} /></button></div>{showInfo && <p className="mt-3 rounded-lg bg-white/[.03] p-2 text-[10px] leading-relaxed text-slate-400">Rows show actual classes; columns show predicted classes. Diagonal cells are correct classifications.</p>}<ConfusionMatrix matrix={model.confusion} /><div className="mt-4 grid grid-cols-2 gap-2 text-[10px]"><span className="flex items-center gap-2 text-slate-400"><span className="h-2 w-2 rounded-sm bg-violet-400/70" /> Correct predictions</span><span className="flex items-center gap-2 text-slate-400"><span className="h-2 w-2 rounded-sm bg-violet-400/20" /> Misclassifications</span></div><div className="mt-5 rounded-xl border border-emerald-400/10 bg-emerald-400/[.04] p-3"><div className="flex items-center gap-2 text-[11px] font-semibold text-emerald-200"><Target size={14} /> Best overall model</div><div className="mt-1 text-[10px] text-slate-500">{metrics.reduce((best, item) => Number(item.f1) > Number(best.f1) ? item : best, metrics[0]).name} has the highest F1 score.</div></div></div>
    </div>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{keys.map((key) => <div key={key} className="rounded-xl border border-white/[.06] bg-[#11182a] p-4"><div className="text-[11px] font-semibold text-slate-300">{key}</div><div className="mt-1 text-[10px] leading-relaxed text-slate-500">{metricInfo[key]}</div></div>)}</div>
  </section>;
}
