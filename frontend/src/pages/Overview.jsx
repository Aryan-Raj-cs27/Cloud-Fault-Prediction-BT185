import { ArrowRight } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Badge, ChartTooltip, Empty, Panel, Stat, Table, Td, fmt, meanStd, pct, riskStatus } from '../components/ui.jsx';
import { useTheme } from '../lib/theme.jsx';

function headline(simulation) {
  if (!simulation?.comparison?.length) return null;
  const rows = simulation.comparison.map((r) => ({ name: r.scheduler, ...meanStd(r.task_interruptions) }));
  const fars = rows.find((r) => r.name === 'fars');
  const best = rows.filter((r) => r.name !== 'fars').sort((a, b) => a.mean - b.mean)[0];
  const sig = simulation.significance?.find((s) => s.B === best.name && s.metric === 'task_interruptions');
  return { rows, fars, best, change: (fars.mean - best.mean) / best.mean, p: sig?.wilcoxon_p };
}

export default function Overview({ monitor, go }) {
  const { palette } = useTheme();
  const { metrics, simulation, session, logs } = monitor;
  const fused = metrics?.models?.find((m) => m.name === 'fused');
  const strict = metrics?.early_warning?.find((m) => m.name === 'fused');
  const h = headline(simulation);
  const valSrc = simulation?.score_sources?.find((r) => r.score_source.startsWith('validation'));
  const ranking = (metrics?.models ?? []).filter((m) => m.pr_auc != null && m.name !== 'always_zero')
    .sort((a, b) => b.pr_auc - a.pr_auc).slice(0, 7);
  const split = metrics?.data;

  return <div className="space-y-4">
    <div className="flex flex-wrap items-end justify-between gap-2">
      <div><h1 className="text-lg font-semibold">Overview</h1><p className="text-xs text-ink3">Early fault prediction on HDFS block logs and its use in fault-aware scheduling. All figures come from the held-out test period.</p></div>
    </div>

    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <Stat label="Early-warning PR-AUC" value={strict ? strict.test_pr_auc.toFixed(3) : '—'} detail={strict ? `Fused risk on blocks still running after 60 s (test). Validation ${strict.val_pr_auc.toFixed(3)}; the standard protocol's ${fused ? fused.pr_auc_text.split('±')[0].trim() : '—'} is inflated by censoring` : 'Run make eval'} />
      <Stat label="Faults flagged early" value={strict?.test_recall != null ? pct(strict.test_recall) : '—'} detail={strict?.test_precision != null ? `still-running test faults · precision ${pct(strict.test_precision)}` : 'Recall at validation threshold'} />
      <Stat label="Task interruptions (FARS)" value={h ? `${h.change > 0 ? '+' : ''}${(100 * h.change).toFixed(0)}%` : '—'} tone={h && h.change < 0 ? 'ok' : undefined}
        detail={h ? `vs ${h.best.name.replace('_', '-')}${h.p != null ? `, p ${h.p < 0.001 ? '< 0.001' : h.p.toFixed(3)}` : ''}${valSrc ? ` · ${valSrc['task_interruptions_vs_best_baseline_%'].toFixed(0)}% with validation-period scores` : ''}` : 'Run make simulate'} />
      <Stat label="Median warning lead time" value={metrics?.lead_time_hours ? fmt(metrics.lead_time_hours.median * 60) : '—'} unit="min"
        detail={metrics?.lead_time_hours?.n_true_positives_alive != null ? `${fmt(metrics.lead_time_hours.n_true_positives_alive)} correct warnings on still-running blocks` : 'Time from warning to end of block session'} />
    </div>

    <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
      <Panel title="Task interruptions by scheduler" subtitle={simulation ? `${simulation.seeds.test.length} test seeds · ${simulation.sim_config.n_nodes} nodes · ${fmt(simulation.sim_config.n_tasks)} tasks · mean ± std` : undefined}
        actions={<button onClick={() => go('scheduling')} className="inline-flex items-center gap-1 text-xs text-accent hover:underline">Details <ArrowRight size={12} /></button>}>
        {h ? <div className="h-56"><ResponsiveContainer><BarChart data={h.rows} margin={{ top: 4, right: 8, left: -14, bottom: 0 }}>
          <CartesianGrid vertical={false} /><XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tickFormatter={(v) => v.replace('_', '-')} /><YAxis tickLine={false} axisLine={false} />
          <Tooltip content={<ChartTooltip format={(p) => `${p.value.toFixed(1)} ± ${p.payload.std.toFixed(1)} interruptions`} />} />
          <Bar animationDuration={600} animationEasing="ease-out" dataKey="mean" name="interruptions" radius={[3, 3, 0, 0]} maxBarSize={36}>{h.rows.map((r) => <Cell key={r.name} fill={r.name === 'fars' ? palette.accent : palette.muted} />)}<ErrorBar dataKey="std" width={4} stroke={palette.ink3} /></Bar>
        </BarChart></ResponsiveContainer></div> : <Empty title="No simulation results yet">Run <span className="font-mono">make simulate</span>, then restart the API.</Empty>}
      </Panel>

      <Panel title="Model ranking · 60 s window" subtitle="Standard-protocol test PR-AUC, inflated by censoring; strict early-warning scores are on the Models page" bodyClass="p-0"
        actions={<button onClick={() => go('models')} className="inline-flex items-center gap-1 text-xs text-accent hover:underline">All models <ArrowRight size={12} /></button>}>
        {ranking.length ? <Table head={['Model', 'Family', { label: 'PR-AUC', right: true }, { label: 'F1', right: true }]} minWidth={360}>
          {ranking.map((m) => <tr key={m.name}><Td mono className={m.name === 'fused' ? 'font-medium text-ink' : 'text-ink2'}>{m.name}</Td><Td className="text-ink3">{m.family}</Td><Td right>{m.pr_auc_text}</Td><Td right className="text-ink2">{m.f1_text}</Td></tr>)}
        </Table> : <div className="p-4"><Empty title="No evaluation results yet">Run <span className="font-mono">make eval</span>, then restart the API.</Empty></div>}
      </Panel>
    </div>

    <div className="grid gap-4 xl:grid-cols-[1fr_1.2fr]">
      <Panel title="Pipeline configuration" bodyClass="p-0">
        <dl className="divide-y divide-line text-xs">
          {[
            ['Blocks (train / val / test)', split ? `${fmt(split.train[0])} / ${fmt(split.val[0])} / ${fmt(split.test[0])}` : '—'],
            ['Fault rate (train / val / test)', split ? `${pct(split.train[1] / split.train[0])} / ${pct(split.val[1] / split.val[0])} / ${pct(split.test[1] / split.test[0])}` : '—'],
            ['Split', 'Chronological 70 / 15 / 15, 30 min gap'],
            ['Fused models', metrics?.best ? `${metrics.best.best_supervised} + ${metrics.best.best_anomaly}` : '—'],
            ['Fusion weight α · threshold', metrics?.fusion?.alpha != null ? `${metrics.fusion.alpha} · ${Number(metrics.fusion.threshold).toFixed(3)}` : '—'],
            ['FARS τ_low · τ_high', simulation ? `${simulation.fars.tau_low} · ${simulation.fars.tau_high}` : '—'],
          ].map(([k, v]) => <div key={k} className="flex justify-between gap-4 px-4 py-2"><dt className="text-ink3">{k}</dt><dd className="num text-right font-mono text-[11.5px] text-ink">{v}</dd></div>)}
        </dl>
      </Panel>

      <Panel title="Live session" subtitle="Held-out blocks scored in this browser session" bodyClass="p-0"
        actions={<button onClick={() => go('inference')} className="inline-flex items-center gap-1 text-xs text-accent hover:underline">Open live inference <ArrowRight size={12} /></button>}>
        {logs.length ? <>
          <div className="grid grid-cols-4 divide-x divide-line border-b border-line text-center text-xs">
            {[['Scored', session.total], ['Faults caught', `${session.tp}/${session.tp + session.fn}`], ['False alarms', session.fp], ['Correct', session.labelled ? pct((session.tp + session.tn) / session.labelled) : '—']].map(([k, v]) => <div key={k} className="px-2 py-2"><div className="text-2xs text-ink3">{k}</div><div className="num text-sm font-semibold">{v}</div></div>)}
          </div>
          <Table head={['Block', 'Verdict', { label: 'Risk', right: true }, 'Truth']} minWidth={360}>
            {logs.slice(0, 5).map((l) => <tr key={l.id}><Td mono className="max-w-[180px] truncate text-ink2">{l.blockId ?? 'pasted'}</Td><Td><Badge status={riskStatus(l.riskLevel)}>{l.status === 'Anomaly' ? 'Fault predicted' : 'Normal'}</Badge></Td><Td right>{l.risk.toFixed(1)}%</Td><Td className="text-ink2">{l.truth ?? '—'}</Td></tr>)}
          </Table>
        </> : <div className="p-4"><Empty title="Nothing scored yet">Open Live inference and start the replay of held-out test blocks.</Empty></div>}
      </Panel>
    </div>
  </div>;
}
