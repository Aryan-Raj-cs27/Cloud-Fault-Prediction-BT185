import { useMemo, useRef, useState } from 'react';
import { Download, Pause, Play, Upload } from 'lucide-react';
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { Badge, Button, ChartTooltip, Empty, Panel, Segmented, Table, Td, riskStatus } from '../components/ui.jsx';
import { WINDOWS, groupLinesByBlock } from '../lib/api.js';
import { useTheme } from '../lib/theme.jsx';

const time = (iso) => new Date(iso).toLocaleTimeString([], { hour12: false });

function exportCsv(rows) {
  const head = ['time', 'block', 'window', 'verdict', 'risk_pct', 'risk_band', 'truth', 'latency_ms'];
  const esc = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  const body = rows.map((r) => [r.timestamp, r.blockId, r.window, r.status, r.risk, r.riskLevel, r.truth, r.latency].map(esc).join(','));
  const url = URL.createObjectURL(new Blob([[head.join(','), ...body].join('\r\n')], { type: 'text/csv' }));
  Object.assign(document.createElement('a'), { href: url, download: `faultaware-predictions-${new Date().toISOString().slice(0, 10)}.csv` }).click();
  URL.revokeObjectURL(url);
}

function RiskTimeline({ logs, threshold }) {
  const { palette } = useTheme();
  const data = [...logs].reverse().slice(-60).map((l, i) => ({ i: i + 1, risk: l.risk, fault: l.truth === 'Anomaly' ? l.risk : null, block: l.blockId }));
  if (!data.length) return <Empty title="No blocks scored yet">Start the replay, or score a block from the input panel.</Empty>;
  return <div className="h-56"><ResponsiveContainer><ComposedChart data={data} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
    <CartesianGrid vertical={false} /><XAxis dataKey="i" tickLine={false} axisLine={false} /><YAxis domain={[0, 100]} unit="%" tickLine={false} axisLine={false} />
    <Tooltip content={<ChartTooltip format={(p) => (p.dataKey === 'risk' ? `risk ${p.value.toFixed(1)}% · ${p.payload.block ?? 'pasted'}` : 'ground truth: fault')} />} />
    {threshold != null && <ReferenceLine y={threshold * 100} stroke={palette.crit} strokeDasharray="4 3" label={{ value: 'threshold', position: 'insideTopRight', fill: palette.ink3, fontSize: 10 }} />}
    <Line dataKey="risk" stroke={palette.accent} strokeWidth={2} dot={false} isAnimationActive={false} />
    <Scatter dataKey="fault" fill={palette.crit} shape="diamond" isAnimationActive={false} />
  </ComposedChart></ResponsiveContainer></div>;
}

function RiskBar({ risk, threshold }) {
  return <div className="relative mt-2 h-2 rounded-full bg-sunken">
    <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${Math.min(100, risk)}%` }} />
    {threshold != null && <div className="absolute -top-1 h-4 w-px bg-crit" style={{ left: `${threshold * 100}%` }} title="decision threshold" />}
  </div>;
}

function Detail({ record }) {
  if (!record) return <Empty title="Select a prediction">Click a row to see how its risk was computed.</Empty>;
  const r = record.raw;
  const rows = [
    ['Supervised probability', `${(r.probability_supervised * 100).toFixed(2)}%`, r.models.supervised],
    ['Calibrated probability', `${(r.probability_calibrated * 100).toFixed(2)}%`, 'isotonic, fitted on validation'],
    ['Anomaly score', `${(r.anomaly_score * 100).toFixed(2)}%`, r.models.anomaly],
    ['Fusion weight α', r.alpha, 'r = α·p + (1 − α)·a'],
    ['Decision threshold', `${(r.threshold * 100).toFixed(2)}%`, 'max-F1 on validation'],
    ['Events observed', r.events_observed, WINDOWS[r.window]?.label],
    ['Unknown templates', r.unknown_templates, 'not seen in training'],
    ['WARN lines observed', r.warnings_observed, ''],
  ];
  return <div className="space-y-3">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <span className="font-mono text-[11.5px] text-ink2">{record.blockId ?? 'pasted block'}</span>
      <div className="flex gap-1.5"><Badge status={riskStatus(record.riskLevel)}>{record.riskLevel} risk</Badge>{record.truth && <Badge status={record.truth === record.status ? 'ok' : 'crit'}>{record.truth === record.status ? 'matches truth' : `truth: ${record.truth}`}</Badge>}</div>
    </div>
    <div><div className="flex items-baseline justify-between"><span className="text-2xs uppercase tracking-wide text-ink3">Fused risk</span><span className="num text-xl font-semibold">{record.risk.toFixed(1)}%</span></div><RiskBar risk={record.risk} threshold={r.threshold} /></div>
    <dl className="divide-y divide-line rounded border border-line text-xs">{rows.map(([k, v, note]) => <div key={k} className="grid grid-cols-[1fr_auto] gap-x-3 px-3 py-1.5"><dt className="text-ink2">{k}{note && <div className="text-2xs text-ink3">{note}</div>}</dt><dd className="num self-center font-mono text-[11.5px]">{v}</dd></div>)}</dl>
    {record.lines?.length > 0 && <div><div className="mb-1 text-2xs uppercase tracking-wide text-ink3">Log lines ({record.lines.length})</div>
      <pre className="max-h-56 overflow-auto rounded border border-line bg-sunken p-2 font-mono text-[10.5px] leading-relaxed">{record.lines.map((line, i) => {
        const flag = / WARN | ERROR |exception|Exception/.test(line);
        return <div key={i} className={flag ? 'text-crit' : 'text-ink2'}>{line}</div>;
      })}</pre></div>}
  </div>;
}

function InputPanel({ source, monitor, onNotice }) {
  const { running, setRunning, replayOne, predictLines, loadSample } = monitor;
  const [text, setText] = useState('');
  const [truth, setTruth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState(null);
  const fileRef = useRef(null);

  if (source === 'replay') return <div className="space-y-3 text-xs text-ink2">
    <p>Each tick fetches a real block from the unseen test period (Nov 10–11, 2008), sends its raw log lines to <span className="font-mono">POST /predict</span>, and records the verdict next to the expert label.</p>
    <div className="flex flex-wrap gap-2">
      <Button variant={running ? 'secondary' : 'primary'} onClick={() => setRunning(!running)}>{running ? <><Pause size={13} /> Pause replay</> : <><Play size={13} /> Start replay (every 2 s)</>}</Button>
      <Button onClick={() => replayOne('Anomaly')}>Score a faulty block</Button>
      <Button onClick={() => replayOne('Normal')}>Score a normal block</Button>
    </div>
  </div>;

  if (source === 'paste') {
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    const example = async (label) => { try { const s = await loadSample(label); setText(s.log_lines.join('\n')); setTruth(s.label); } catch { onNotice('Could not load an example from the API.'); } };
    const run = async () => { setBusy(true); const r = await predictLines(lines, truth); setBusy(false); if (!r) onNotice('Prediction failed: check the log format and the API.'); };
    return <div className="space-y-2">
      <textarea aria-label="Raw HDFS log lines for one block" value={text} onChange={(e) => { setText(e.target.value); setTruth(null); }} rows={9} spellCheck={false}
        placeholder="081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
        className="w-full resize-y rounded border border-line bg-sunken p-2 font-mono text-[10.5px] leading-relaxed text-ink outline-none focus:border-accent" />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex gap-2"><Button onClick={() => example('Anomaly')}>Load faulty example</Button><Button onClick={() => example('Normal')}>Load normal example</Button></div>
        <Button variant="primary" disabled={busy || !lines.length} onClick={run}>{busy ? 'Scoring…' : `Score ${lines.length} line${lines.length === 1 ? '' : 's'}`}</Button>
      </div>
      {truth && <p className="text-2xs text-ink3">Example label (not sent to the model): {truth}</p>}
    </div>;
  }

  const runFile = async () => {
    setBusy(true);
    try {
      const groups = groupLinesByBlock(await file.text());
      if (!groups.size) throw new Error('No block ids (blk_…) found in this file.');
      let n = 0;
      for (const [block, lines] of [...groups].slice(0, 500)) {
        if (!(await predictLines(lines, null, block))) throw new Error(`Stopped after ${n} blocks: the API returned an error.`);
        n += 1;
      }
      onNotice(`Scored ${n} block${n === 1 ? '' : 's'}${groups.size > 500 ? ` (first 500 of ${groups.size})` : ''}.`);
    } catch (e) { onNotice(e.message); } finally { setBusy(false); }
  };
  return <div className="space-y-2 text-xs text-ink2">
    <p>Upload an HDFS log excerpt. Lines are grouped by block id in the browser and each block is scored separately (max 500 blocks).</p>
    <input ref={fileRef} type="file" accept=".log,.txt,text/plain" className="hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
    <div className="flex flex-wrap items-center gap-2"><Button onClick={() => fileRef.current?.click()}><Upload size={13} /> {file ? file.name : 'Choose file'}</Button><Button variant="primary" disabled={!file || busy} onClick={runFile}>{busy ? 'Scoring…' : 'Score file'}</Button></div>
  </div>;
}

export default function LiveInference({ monitor }) {
  const { obsWindow, setObsWindow, logs, selected, setSelectedId, session, error } = monitor;
  const [source, setSource] = useState('replay');
  const [filter, setFilter] = useState('all');
  const [notice, setNotice] = useState('');
  const rows = useMemo(() => logs.filter((l) => (filter === 'flagged' ? l.status === 'Anomaly'
    : filter === 'errors' ? l.truth && l.truth !== l.status : true)), [logs, filter]);
  const threshold = logs[0]?.raw?.threshold;

  return <div className="space-y-4">
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div><h1 className="text-lg font-semibold">Live inference</h1><p className="text-xs text-ink3">Score HDFS blocks from their raw log lines with the fused risk model.</p></div>
      <div className="flex flex-wrap items-center gap-2 text-2xs text-ink3">Horizon <Segmented label="Prediction horizon" value={obsWindow} onChange={setObsWindow} options={Object.entries(WINDOWS).map(([value, w]) => ({ value, label: w.short }))} /></div>
    </div>
    {(error || notice) && <div className="rounded border border-line bg-surface px-3 py-2 text-xs text-ink2">{error || notice}</div>}

    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(340px,1fr)]">
      <div className="space-y-4">
        <Panel title="Risk timeline" subtitle={`Last 60 scored blocks · ${WINDOWS[obsWindow].label} · diamonds mark blocks whose expert label is “fault”`}><RiskTimeline logs={logs} threshold={threshold} /></Panel>
        <Panel title="Predictions" bodyClass="p-0"
          subtitle={session.labelled ? `Against ground truth: ${session.tp} caught, ${session.fn} missed, ${session.fp} false alarms, ${session.tn} correct normals` : 'Ground truth is shown for replayed test blocks'}
          actions={<><Segmented label="Filter" value={filter} onChange={setFilter} options={[{ value: 'all', label: 'All' }, { value: 'flagged', label: 'Flagged' }, { value: 'errors', label: 'Errors' }]} /><Button onClick={() => exportCsv(rows)} disabled={!rows.length}><Download size={13} /> CSV</Button></>}>
          {rows.length ? <div className="max-h-[420px] overflow-y-auto"><Table head={['Time', 'Block', 'Verdict', { label: 'Risk', right: true }, 'Truth', { label: 'Latency', right: true }]} minWidth={620}>
            {rows.map((l) => <tr key={l.id} onClick={() => setSelectedId(l.id)} className={`cursor-pointer ${selected?.id === l.id ? 'bg-accent/[.07]' : 'hover:bg-sunken'}`}>
              <Td className="num whitespace-nowrap text-ink3">{time(l.timestamp)}</Td>
              <Td mono className="max-w-[200px] truncate text-ink2">{l.blockId ?? 'pasted'}</Td>
              <Td><Badge status={riskStatus(l.riskLevel)}>{l.status === 'Anomaly' ? 'Fault predicted' : 'Normal'}</Badge></Td>
              <Td right>{l.risk.toFixed(1)}%</Td>
              <Td className={l.truth && l.truth !== l.status ? 'font-medium text-crit' : 'text-ink2'}>{l.truth ?? '—'}</Td>
              <Td right className="text-ink3">{l.latency} ms</Td>
            </tr>)}
          </Table></div> : <div className="p-4"><Empty title="No predictions match this filter" /></div>}
        </Panel>
      </div>
      <div className="space-y-4">
        <Panel title="Input" actions={<Segmented label="Input source" value={source} onChange={setSource} options={[{ value: 'replay', label: 'Replay' }, { value: 'paste', label: 'Paste' }, { value: 'file', label: 'File' }]} />}>
          <InputPanel source={source} monitor={monitor} onNotice={setNotice} />
        </Panel>
        <Panel title="Prediction detail"><Detail record={selected} /></Panel>
      </div>
    </div>
  </div>;
}
