import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ChartTooltip, Empty, Panel, Stat, Table, Td, fmt, meanStd } from '../components/ui.jsx';
import { useTheme } from '../lib/theme.jsx';

const METRICS = {
  task_interruptions: ['Task interruptions', 'count', 1],
  wasted_compute_min: ['Wasted compute', 'min', 0],
  lost_node_min: ['Lost node time', 'min', 0],
  sla_violation_rate: ['SLA violation rate', '', 3],
  mean_response_min: ['Mean response time', 'min', 1],
  migrations: ['Migrations', 'count', 0],
  energy_kwh: ['Energy', 'kWh', 1],
};

function Heatmap({ trace, title, metrics, palette }) {
  const cols = trace.slice(0, 200);
  if (!cols.length) return null;
  const n = Math.max(...cols.map((c) => Object.keys(c.risk).length + c.down.length));
  const cw = 3; const ch = 5;
  const shade = (r) => palette.seq[Math.min(palette.seq.length - 1, Math.floor(r * palette.seq.length))];
  return <div className="min-w-0">
    <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2"><span className="font-mono text-xs font-medium">{title}</span><span className="num text-2xs text-ink3">{metrics.task_interruptions} interruptions · {fmt(metrics.wasted_compute_min)} min wasted · {metrics.migrations} migrations</span></div>
    <div className="overflow-x-auto rounded border border-line bg-sunken p-1.5">
      <svg width={cols.length * cw} height={n * ch} role="img" aria-label={`${title}: node risk over time`} style={{ display: 'block' }}>
        {cols.map((c, x) => Array.from({ length: n }, (_, node) => {
          const down = c.down.includes(node);
          return <rect key={`${x}-${node}`} x={x * cw} y={node * ch} width={cw - 0.4} height={ch - 0.6} fill={down ? palette.crit : shade(c.risk[node] ?? 0)}>
            <title>{`t = ${c.t} min · node ${node} · ${down ? 'down' : `risk ${(c.risk[node] ?? 0).toFixed(2)}`}${c.degraded.includes(node) ? ' · degrading (hidden from scheduler)' : ''}`}</title></rect>;
        }))}
      </svg>
    </div>
  </div>;
}

export default function Scheduling({ monitor }) {
  const { palette } = useTheme();
  const { simulation } = monitor;
  const [metric, setMetric] = useState('task_interruptions');
  const data = useMemo(() => (simulation?.comparison ?? []).map((r) => ({ name: r.scheduler, ...meanStd(r[metric]) })), [simulation, metric]);
  if (!simulation?.comparison?.length) return <div className="space-y-4"><h1 className="text-lg font-semibold">FARS scheduling</h1><Empty title="No simulation results available">Run <span className="font-mono">make simulate</span>, then restart the API.</Empty></div>;

  const cfg = simulation.sim_config;
  const tested = ['task_interruptions', 'wasted_compute_min', 'lost_node_min', 'sla_violation_rate', 'mean_response_min'];
  const sig = simulation.significance.filter((s) => s.metric === (tested.includes(metric) ? metric : 'task_interruptions'));
  const [label, unit, digits] = METRICS[metric];
  const fars = data.find((d) => d.name === 'fars');
  const ll = simulation.comparison.find((r) => r.scheduler === 'least_loaded');
  const noise = simulation.noise.map((r) => ({ noise: r.noise, ...meanStd(r.lost_node_min) }));
  const llLost = meanStd(ll?.lost_node_min).mean;

  return <div className="space-y-4">
    <div><h1 className="text-lg font-semibold">FARS scheduling</h1>
      <p className="max-w-4xl text-xs text-ink3">Fault-Aware Risk-based Scheduler on a SimPy cluster ({cfg.n_nodes} nodes, {fmt(cfg.n_tasks)} tasks, Weibull node failures, {Math.round(100 * (1 - cfg.unpredictable_fraction))}% preceded by a hidden degradation window). Every {cfg.tick_min} min the scheduler sees one risk score per node, drawn from the real predictor's scores for blocks still running after the 60 s window (test period; the validation period is shown below as a pessimistic case). Thresholds τ were tuned on {simulation.seeds.validation.length} validation seeds; results use {simulation.seeds.test.length} separate test seeds with identical workloads and faults across schedulers.</p></div>

    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <Stat label="τ_low · τ_high" value={`${simulation.fars.tau_low} · ${simulation.fars.tau_high}`} detail="avoid placement · migrate / checkpoint" />
      {['task_interruptions', 'wasted_compute_min', 'lost_node_min'].map((m) => {
        const s = simulation.significance.find((x) => x.B === 'least_loaded' && x.metric === m);
        return s && <Stat key={m} label={`${METRICS[m][0]} vs least-loaded`} value={`${s['rel_change_%'] > 0 ? '+' : ''}${s['rel_change_%'].toFixed(1)}%`} tone={s['rel_change_%'] < 0 ? 'ok' : 'crit'} detail={`Wilcoxon p ${s.wilcoxon_p < 0.001 ? '< 0.001' : s.wilcoxon_p.toFixed(3)} · ${simulation.seeds.test.length} paired seeds`} />;
      })}
    </div>

    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(300px,1fr)]">
      <Panel title={`${label} by scheduler`} subtitle="mean ± std over test seeds" actions={<select aria-label="Metric" value={metric} onChange={(e) => setMetric(e.target.value)} className="max-w-full rounded border border-line bg-surface px-2 py-1 text-xs">{Object.entries(METRICS).map(([k, [l]]) => <option key={k} value={k}>{l}</option>)}</select>}>
        <div className="h-64"><ResponsiveContainer><BarChart data={data} margin={{ top: 4, right: 8, left: -6, bottom: 0 }}>
          <CartesianGrid vertical={false} /><XAxis dataKey="name" interval={0} tickLine={false} axisLine={false} tickFormatter={(v) => v.replace('_', '-')} /><YAxis tickLine={false} axisLine={false} />
          <Tooltip content={<ChartTooltip format={(p) => `${p.value.toFixed(digits)} ± ${p.payload.std.toFixed(digits)} ${unit}`} />} />
          {fars && <ReferenceLine y={fars.mean} stroke={palette.accent} strokeDasharray="3 3" />}
          <Bar animationDuration={600} animationEasing="ease-out" dataKey="mean" radius={[3, 3, 0, 0]} maxBarSize={40}>{data.map((d) => <Cell key={d.name} fill={d.name === 'fars' ? palette.accent : palette.muted} />)}<ErrorBar dataKey="std" width={4} stroke={palette.ink3} /></Bar>
        </BarChart></ResponsiveContainer></div>
      </Panel>
      <Panel title="FARS vs each baseline" subtitle={`Paired Wilcoxon signed-rank · ${sig[0]?.metric?.replaceAll('_', ' ')}`} bodyClass="p-0">
        <Table head={['Baseline', { label: 'Baseline mean', right: true }, { label: 'FARS mean', right: true }, { label: 'Change', right: true }, { label: 'p', right: true }]} minWidth={380}>
          {sig.map((s) => <tr key={s.B}><Td mono>{s.B}</Td><Td right className="text-ink2">{s.mean_B.toFixed(2)}</Td><Td right>{s.mean_A.toFixed(2)}</Td><Td right className={s['rel_change_%'] < 0 ? 'text-ok' : 'text-crit'}>{s['rel_change_%'] > 0 ? '+' : ''}{s['rel_change_%'].toFixed(1)}%</Td><Td right className="text-ink2">{s.wilcoxon_p < 0.001 ? '< 0.001' : s.wilcoxon_p.toFixed(3)}</Td></tr>)}
        </Table>
      </Panel>
    </div>

    <Panel title="Cluster replay · test seed 0" subtitle="Rows are nodes, columns are 5-minute ticks. Shade = risk score seen by the scheduler (darker = higher); red = node down. Hover a cell for details."
      actions={<div className="flex items-center gap-1 text-2xs text-ink3">low {palette.seq.map((c) => <span key={c} className="h-2.5 w-3" style={{ background: c }} />)} high <span className="ml-2 h-2.5 w-3" style={{ background: palette.crit }} /> down</div>}>
      <div className="grid gap-4 lg:grid-cols-2">{['fars', 'least_loaded'].map((name) => simulation.replay?.[name] && <Heatmap key={name} title={name} trace={simulation.replay[name].trace} metrics={simulation.replay[name].metrics} palette={palette} />)}</div>
    </Panel>

    {simulation.score_sources?.length > 0 && <Panel title="Which risk scores drive FARS" subtitle="Same test seeds, workloads, faults and τ; only the distribution the node risk scores are drawn from changes. Both use blocks still running after the 60 s window. Change is relative to the best baseline." bodyClass="p-0">
      <Table head={['Risk scores from', { label: 'Interruptions', right: true }, { label: 'Change', right: true }, { label: 'Lost node-min', right: true }, { label: 'Change', right: true }, { label: 'Migrations', right: true }]} minWidth={720}>
        {simulation.score_sources.map((r) => {
          const tone = (v) => (v < 0 ? 'text-ok' : 'text-crit');
          const pc = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
          return <tr key={r.score_source}><Td className="text-ink2">{r.score_source}</Td>
            <Td right>{r.task_interruptions}</Td><Td right className={tone(r['task_interruptions_vs_best_baseline_%'])}>{pc(r['task_interruptions_vs_best_baseline_%'])}</Td>
            <Td right>{r.lost_node_min}</Td><Td right className={tone(r['lost_node_min_vs_best_baseline_%'])}>{pc(r['lost_node_min_vs_best_baseline_%'])}</Td>
            <Td right className="text-ink2">{r.migrations}</Td></tr>;
        })}
      </Table>
    </Panel>}

    <div className="grid gap-4 xl:grid-cols-3">
      <Panel title="Simulator sanity checks" subtitle="An oracle predictor should approach zero loss; a random one should not help" bodyClass="p-0">
        <Table head={['Variant', { label: 'Interruptions', right: true }, { label: 'Lost node-min', right: true }]} minWidth={300}>
          {simulation.sanity.map((r) => <tr key={r.scheduler}><Td mono>{r.scheduler}</Td><Td right>{r.task_interruptions}</Td><Td right className="text-ink2">{r.lost_node_min}</Td></tr>)}
        </Table>
      </Panel>
      <Panel title="Ablations" subtitle="τ re-tuned on validation for each variant" bodyClass="p-0">
        <Table head={['Variant', { label: 'Interruptions', right: true }, { label: 'Lost node-min', right: true }]} minWidth={300}>
          {simulation.ablations.map((r) => <tr key={r.variant}><Td className="text-ink2">{r.variant}</Td><Td right>{r.task_interruptions}</Td><Td right className="text-ink2">{r.lost_node_min}</Td></tr>)}
        </Table>
      </Panel>
      <Panel title="Noise sensitivity" subtitle="Share of risk scores replaced by random noise">
        <div className="h-48"><ResponsiveContainer><LineChart data={noise} margin={{ top: 6, right: 8, left: -6, bottom: 0 }}>
          <CartesianGrid vertical={false} /><XAxis dataKey="noise" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} />
          <Tooltip content={<ChartTooltip label="" format={(p) => `η = ${p.payload.noise}: ${fmt(p.value)} ± ${fmt(p.payload.std)} lost node-min`} />} />
          {Number.isFinite(llLost) && <ReferenceLine y={llLost} stroke={palette.ink3} strokeDasharray="4 3" label={{ value: 'least-loaded', position: 'insideTopLeft', fill: palette.ink3, fontSize: 10 }} />}
          <Line dataKey="mean" stroke={palette.accent} strokeWidth={2} dot={{ r: 3 }} animationDuration={600} animationEasing="ease-out" />
        </LineChart></ResponsiveContainer></div>
      </Panel>
    </div>
  </div>;
}
