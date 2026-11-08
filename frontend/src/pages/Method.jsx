import { ChevronRight } from 'lucide-react';
import { Panel, Table, Td, fmt, pct } from '../components/ui.jsx';

const STAGES = [
  ['Raw logs', 'HDFS_v1 · 11.2 M lines'],
  ['Parse', 'mask ids, IPs, paths → 54 templates'],
  ['Sessions', '575 k labelled blocks'],
  ['Split', 'chronological + 30 min gap'],
  ['Features', 'counts + sequences, first W s only'],
  ['Models', 'ML · DL · anomaly detectors'],
  ['Fusion', 'isotonic calibration, α-weighted'],
  ['FARS', 'placement, migration, checkpoints'],
];

const RULES = [
  ['r ≥ τ_high', 'Migrate running tasks to the best low-risk node; if none fits, checkpoint now at the elevated frequency.'],
  ['τ_low ≤ r < τ_high', 'Place no new tasks on the node while alternatives exist; running tasks checkpoint more often.'],
  ['r < τ_low', 'Normal placement by score = w₁·free capacity − w₂·risk − w₃·migration cost.'],
];

const SAFEGUARDS = [
  'Sessions are split by start time; blocks starting within 30 min after a boundary are dropped (tested: max(train) + gap < min(val)).',
  'The template vocabulary and every scaler are fitted on the training split only; unseen templates map to <UNK>.',
  'Features use only events inside the observation window (tested by mutating later events).',
  'SMOTE is applied to training data only; thresholds, calibration, α and τ are tuned on validation.',
  'The test split is scored once, by scripts/evaluate.py. Full-session scores near 1.0 were investigated (decision D-010).',
  'The trace ends 2 h after the test split starts, so most test faults end inside the window. A separate strict early-warning table scores only blocks still running, and the simulator samples only those scores (decision D-013).',
];

const ASSUMPTIONS = [
  'Node failures follow a Weibull renewal process; 20 % arrive without any precursor and cannot be predicted.',
  'A degrading node yields risk scores drawn from the real model’s scores for faulty blocks, one draw per episode (conservative).',
  'Checkpoints cost 1 min; migration takes 30 s and preserves progress; restarts add a 2 min penalty.',
  'Workload is synthetic (Poisson arrivals, lognormal durations) because HDFS logs contain no task trace.',
];

const TEAM = [
  ['Aryan Raj', 'CS-2341005', 'Core ML/DL & backend'],
  ['Dheeraj Kumar', 'CS-2341062', 'Data engineering'],
  ['Navneet Kumar', 'CS-2341055', 'Quality assurance & testing'],
  ['Priyansh Kumar', 'CS-2341080', 'Frontend & visualisation'],
];

export default function Method({ monitor }) {
  const split = monitor.metrics?.data;
  return <div className="space-y-4">
    <div><h1 className="text-lg font-semibold">Data &amp; method</h1><p className="text-xs text-ink3">BT-185 · Fault-Aware Machine Learning and Deep Learning-Based Algorithm for Cloud Architecture</p></div>

    <Panel title="Pipeline">
      <ol className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
        {STAGES.map(([name, note], i) => <li key={name} className="relative rounded border border-line bg-sunken px-3 py-2">
          <div className="text-2xs text-ink3">Step {i + 1}</div><div className="text-xs font-medium">{name}</div><div className="mt-0.5 text-2xs leading-snug text-ink3">{note}</div>
          {i < STAGES.length - 1 && <ChevronRight size={12} className="absolute -right-[9px] top-1/2 hidden -translate-y-1/2 text-ink3 xl:block" aria-hidden />}
        </li>)}
      </ol>
    </Panel>

    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Prediction task">
        <div className="space-y-2 text-xs leading-relaxed text-ink2">
          <p><span className="font-medium text-ink">Sample.</span> One HDFS block session: every log line that mentions the block, in time order.</p>
          <p><span className="font-medium text-ink">Label.</span> 1 if the block is marked anomalous in the Loghub expert labels (Xu et al., SOSP 2009), else 0.</p>
          <p><span className="font-medium text-ink">Early warning.</span> The model sees only the first <span className="font-mono">W</span> seconds after the block is allocated (W = 60 s primary, 10 s secondary) and must predict the label. In training and validation, faulty blocks typically live for hours, so a correct warning can arrive long before the session ends (the lead time).</p>
          <p><span className="font-medium text-ink">Metric.</span> PR-AUC on the failure class, plus precision, recall, F1 and FPR at a validation-chosen threshold.</p>
        </div>
      </Panel>
      <Panel title="Chronological split" bodyClass="p-0">
        <Table head={['Split', { label: 'Blocks', right: true }, { label: 'Faulty', right: true }, { label: 'Fault rate', right: true }]} minWidth={320}>
          {split ? ['train', 'val', 'test'].map((s) => <tr key={s}><Td mono>{s}</Td><Td right>{fmt(split[s][0])}</Td><Td right>{fmt(split[s][1])}</Td><Td right>{pct(split[s][1] / split[s][0], 2)}</Td></tr>)
            : <tr><Td className="text-ink3" colSpan={4}>Available after make eval</Td></tr>}
        </Table>
        <p className="border-t border-line px-4 py-2 text-2xs text-ink3">The lower fault rate in the test period is a genuine shift over time; a random split would hide it.</p>
      </Panel>
    </div>

    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Leakage safeguards">
        <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-ink2">{SAFEGUARDS.map((s) => <li key={s}>{s}</li>)}</ul>
      </Panel>
      <Panel title="FARS decision rules" subtitle="Risk r is the EWMA-smoothed fused score of each node" bodyClass="p-0">
        <Table head={['Condition', 'Action']} minWidth={360}>{RULES.map(([c, a]) => <tr key={c}><Td mono className="whitespace-nowrap">{c}</Td><Td className="text-ink2">{a}</Td></tr>)}</Table>
      </Panel>
    </div>

    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Simulator assumptions"><ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-ink2">{ASSUMPTIONS.map((s) => <li key={s}>{s}</li>)}</ul></Panel>
      <Panel title="Team" subtitle="Guide: Ms. Niharika Chaudhary" bodyClass="p-0">
        <Table head={['Member', 'Roll no.', 'Role']} minWidth={360}>{TEAM.map(([n, r, role]) => <tr key={r}><Td>{n}</Td><Td mono className="text-ink2">{r}</Td><Td className="text-ink2">{role}</Td></tr>)}</Table>
      </Panel>
    </div>
  </div>;
}
