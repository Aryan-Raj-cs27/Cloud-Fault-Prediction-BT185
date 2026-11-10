import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ChartTooltip, Empty, Panel, Segmented, Stat, Table, Td, fmt, pct } from '../components/ui.jsx';
import { WINDOWS } from '../lib/api.js';
import { FAMILY_SLOT, useTheme } from '../lib/theme.jsx';

function Confusion({ matrix, palette }) {
  if (!Array.isArray(matrix)) return <Empty title="No confusion matrix stored for this model" />;
  const rowTotals = matrix.map((r) => r[0] + r[1]);
  const cell = (v, row, key) => {
    const share = v / Math.max(1, rowTotals[row]);
    const idx = Math.min(palette.seq.length - 1, Math.round(share * (palette.seq.length - 1)));
    const dark = idx >= 4;
    return <div key={key} className="rounded px-2 py-3 text-center" style={{ background: palette.seq[idx], color: dark ? '#fff' : '#14171c' }}>
      <div className="num text-sm font-semibold">{fmt(v)}</div><div className="num text-2xs opacity-80">{pct(share)} of row</div></div>;
  };
  return <div className="grid grid-cols-[84px_1fr_1fr] gap-1 text-2xs">
    <div /><div className="pb-1 text-center text-ink3">Predicted normal</div><div className="pb-1 text-center text-ink3">Predicted fault</div>
    <div className="self-center text-ink3">Actual normal</div>{matrix[0].map((v, i) => cell(v, 0, `a${i}`))}
    <div className="self-center text-ink3">Actual fault</div>{matrix[1].map((v, i) => cell(v, 1, `b${i}`))}
  </div>;
}

export default function Models({ monitor }) {
  const { palette } = useTheme();
  const { metrics, loadMetrics } = monitor;
  const [selected, setSelected] = useState('fused');
  if (!metrics?.models?.length) return <div className="space-y-4"><h1 className="text-lg font-semibold">Models</h1><Empty title="No evaluation results available">Run <span className="font-mono">make train</span> and <span className="font-mono">make eval</span>, then restart the API.</Empty></div>;

  const rows = metrics.models.filter((m) => m.name !== 'always_zero');
  const model = rows.find((m) => m.name === selected) ?? rows[0];
  const byName = Object.fromEntries(rows.map((m) => [m.name, m]));
  const chart = rows.filter((m) => m.pr_auc != null).sort((a, b) => b.pr_auc - a.pr_auc);
  const families = Object.keys(FAMILY_SLOT).filter((f) => chart.some((m) => m.family === f));
  const color = (family) => palette.series[FAMILY_SLOT[family] ?? 7];
  const best = metrics.best;
  const baseRate = metrics.data ? metrics.data.test[1] / metrics.data.test[0] : null;

  return <div className="space-y-4">
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div><h1 className="text-lg font-semibold">Models</h1><p className="text-xs text-ink3">Held-out test period · mean ± std over 5 seeds · decision thresholds chosen on validation. Accuracy is not reported because faults are rare.</p></div>
      <Segmented label="Observation window" value={metrics.window} onChange={(w) => loadMetrics(w)} options={metrics.windows.map((w) => ({ value: w, label: WINDOWS[w]?.label ?? w }))} />
    </div>

    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      {[['Fused risk', 'fused'], ['Best classical', best.best_classical], ['Best deep learning', best.best_deep], ['Best unsupervised', best.best_anomaly]].map(([label, name]) => byName[name] &&
        <Stat key={label} label={`${label} · PR-AUC`} value={byName[name].pr_auc_text.split('±')[0].trim()} unit={`± ${byName[name].pr_auc_text.split('±')[1]?.trim()}`} detail={<span className="font-mono">{name}</span>} />)}
    </div>

    <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.6fr)_minmax(320px,1fr)]">
      <Panel title="Model comparison" subtitle="Click a row to inspect it" bodyClass="p-0">
        <Table head={['Model', 'Family', { label: 'PR-AUC', right: true }, { label: 'ROC-AUC', right: true }, { label: 'F1', right: true }, { label: 'Precision', right: true }, { label: 'Recall', right: true }, { label: 'FPR', right: true }, { label: 'ms / block', right: true }]} minWidth={780}>
          {rows.map((m) => <tr key={m.name} onClick={() => setSelected(m.name)} className={`cursor-pointer ${model.name === m.name ? 'bg-accent/[.07]' : 'hover:bg-sunken'}`}>
            <Td mono className="text-ink">{m.name}</Td>
            <Td className="whitespace-nowrap"><span className="inline-flex items-center gap-1.5 text-ink2"><span className="h-2 w-2 rounded-sm" style={{ background: color(m.family) }} />{m.family}</span></Td>
            <Td right className="font-medium">{m.pr_auc_text}</Td><Td right>{m.roc_auc?.toFixed(3) ?? '—'}</Td><Td right>{m.f1_text}</Td>
            <Td right className="text-ink2">{pct(m.precision)}</Td><Td right className="text-ink2">{pct(m.recall)}</Td><Td right className="text-ink2">{pct(m.fpr, 2)}</Td>
            <Td right className="text-ink3">{typeof m.latency_ms === 'number' ? m.latency_ms.toFixed(3) : '—'}</Td>
          </tr>)}
        </Table>
      </Panel>
      <Panel title="Confusion matrix" subtitle={<span><span className="font-mono">{model.name}</span> · seed 0 · validation threshold</span>}><Confusion matrix={model.confusion_matrix} palette={palette} /></Panel>
    </div>

    <Panel title="Test PR-AUC by model" subtitle={baseRate ? `A random ranking scores ≈ the fault base rate (${pct(baseRate)})` : undefined}
      actions={<div className="flex flex-wrap gap-3 text-2xs text-ink2">{families.map((f) => <span key={f} className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: color(f) }} />{f}</span>)}</div>}>
      <div style={{ height: Math.max(200, chart.length * 24) }}><ResponsiveContainer><BarChart data={chart} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }}>
        <CartesianGrid horizontal={false} /><XAxis type="number" domain={[0, 1]} tickLine={false} axisLine={false} /><YAxis type="category" dataKey="name" width={130} tickLine={false} axisLine={false} tick={{ fontFamily: 'IBM Plex Mono', fontSize: 11 }} />
        <Tooltip content={<ChartTooltip format={(p) => `PR-AUC ${p.payload.pr_auc_text}`} />} />
        <Bar animationDuration={600} animationEasing="ease-out" dataKey="pr_auc" radius={[0, 3, 3, 0]} maxBarSize={14}>{chart.map((m) => <Cell key={m.name} fill={color(m.family)} />)}</Bar>
      </BarChart></ResponsiveContainer></div>
    </Panel>

    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Calibration and lead time" bodyClass="p-0">
        <dl className="divide-y divide-line text-xs">
          {[
            ['Brier score, raw probability', metrics.calibration?.brier_raw?.toFixed(4)],
            ['Brier score, isotonic-calibrated', metrics.calibration?.brier_calibrated?.toFixed(4)],
            ['Brier score, fused risk', metrics.calibration?.brier_fused?.toFixed(4)],
            ['Fusion weight α', metrics.fusion?.alpha],
            ['Correct early warnings', metrics.lead_time_hours ? fmt(metrics.lead_time_hours.n_true_positives) : null],
            ['… of which still running', metrics.lead_time_hours?.n_true_positives_alive != null ? fmt(metrics.lead_time_hours.n_true_positives_alive) : null],
            ['Lead time (running), median (IQR)', metrics.lead_time_hours ? `${metrics.lead_time_hours.median.toFixed(1)} h (${metrics.lead_time_hours.p25.toFixed(1)}–${metrics.lead_time_hours.p75.toFixed(1)} h)` : null],
          ].map(([k, v]) => <div key={k} className="flex justify-between px-4 py-2"><dt className="text-ink3">{k}</dt><dd className="num font-mono text-[11.5px]">{v ?? '—'}</dd></div>)}
        </dl>
      </Panel>
      <Panel title="What drives XGBoost predictions" subtitle="Mean |SHAP| on a test sample (60 s window)" bodyClass="p-0">
        {metrics.shap?.global_top?.length ? <Table head={['Feature', { label: 'mean |SHAP|', right: true }]} minWidth={360}>
          {metrics.shap.global_top.slice(0, 8).map((f) => <tr key={f.feature}><Td mono className="max-w-[420px] truncate text-ink2" title={f.feature}>{f.feature.replace(/^count\[/, '').replace(/\]$/, '')}</Td><Td right>{f.mean_abs_shap.toFixed(3)}</Td></tr>)}
        </Table> : <div className="p-4"><Empty title="SHAP values are computed for the 60 s window only" /></div>}
      </Panel>
    </div>

    {metrics.early_warning?.length > 0 && <Panel title="Strict early warning" subtitle={`Only blocks still alive when the ${metrics.window} window closes, so the fault has not happened yet (${metrics.early_warning[0].test_alive} in test). The trace end truncates test-period lifetimes, so the main table partly measures detection.`} bodyClass="p-0">
      <Table head={['Model', { label: 'Validation PR-AUC', right: true }, { label: 'Test PR-AUC', right: true }, { label: 'Test F1', right: true }]} minWidth={520}>
        {metrics.early_warning.filter((r) => !['always_zero', 'random_rate'].includes(r.name)).map((r) => <tr key={r.name}><Td mono>{r.name}</Td><Td right>{fmt(r.val_pr_auc, 3)}</Td><Td right>{fmt(r.test_pr_auc, 3)}</Td><Td right>{fmt(r.test_f1, 3)}</Td></tr>)}
      </Table>
    </Panel>}

    {metrics.significance?.length > 0 && <Panel title="Paired significance tests" subtitle="Test PR-AUC per seed; with 5 seeds the smallest two-sided Wilcoxon p is 0.0625, so the paired t-test is shown as well" bodyClass="p-0">
      <Table head={['A', 'B', { label: 'Mean A', right: true }, { label: 'Mean B', right: true }, { label: 'Wilcoxon p', right: true }, { label: 't-test p', right: true }, 'A, seed-0 95% CI']} minWidth={640}>
        {metrics.significance.map((s) => <tr key={`${s.A}-${s.B}`}><Td mono>{s.A}</Td><Td mono className="text-ink2">{s.B}</Td><Td right>{s.mean_A.toFixed(4)}</Td><Td right>{s.mean_B.toFixed(4)}</Td><Td right>{s.wilcoxon_p.toFixed(4)}</Td><Td right>{s.ttest_p < 1e-4 ? '< 1e-4' : s.ttest_p.toFixed(4)}</Td><Td mono className="text-ink3">{s.A_seed0_95CI}</Td></tr>)}
      </Table>
    </Panel>}
  </div>;
}
