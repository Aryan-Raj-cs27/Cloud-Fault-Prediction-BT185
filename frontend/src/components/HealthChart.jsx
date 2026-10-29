import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Activity } from 'lucide-react';
import { useState } from 'react';

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return <div className="rounded-xl border border-white/10 bg-[#151d31] px-3 py-2.5 shadow-xl"><p className="text-[10px] text-slate-400">{label}</p><p className="mt-1 text-xs font-semibold text-slate-100">Health score <span className="ml-2 text-emerald-300">{payload[0].value}%</span></p></div>;
}

export default function HealthChart({ points, preview }) {
  const [range, setRange] = useState(10);
  const visiblePoints = points.slice(-range);

  return <section className="rounded-2xl border border-white/[.07] bg-[#11182a] p-5 shadow-card sm:p-6">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><div className="flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-100"><Activity size={16} className="text-violet-300" /> Server health timeline{preview && <span className="rounded-md bg-sky-400/[.08] px-1.5 py-0.5 text-[9px] font-medium text-sky-300">PREVIEW</span>}</div><p className="mt-1 text-xs text-slate-500">{preview ? 'Sample values · live results replace these after inference' : 'Health score across recent inference windows'}</p></div>
      <label className="flex items-center gap-2 rounded-lg border border-white/[.08] bg-white/[.025] px-3 py-2 text-[10px] text-slate-500">Show
        <select value={range} onChange={(event) => setRange(Number(event.target.value))} aria-label="Timeline observation range" className="bg-transparent font-medium text-slate-300 outline-none">
          <option value="10">10 observations</option><option value="20">20 observations</option><option value="30">30 observations</option>
        </select>
      </label>
    </div>
    <div className="mt-6 h-[260px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={visiblePoints} margin={{ top: 8, right: 6, left: -18, bottom: 0 }}>
          <defs><linearGradient id="healthFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#8b7cff" stopOpacity={0.28} /><stop offset="95%" stopColor="#8b7cff" stopOpacity={0} /></linearGradient></defs>
          <CartesianGrid strokeDasharray="3 5" vertical={false} />
          <XAxis dataKey="time" tick={{ fill: '#69758d', fontSize: 10 }} axisLine={false} tickLine={false} minTickGap={32} />
          <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={{ fill: '#69758d', fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip />} />
          <Area type="monotone" dataKey="health" stroke="#9b8cff" strokeWidth={2.5} fill="url(#healthFill)" activeDot={{ r: 5, fill: '#c4b5fd', stroke: '#11182a', strokeWidth: 3 }} isAnimationActive animationDuration={700} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
    <div className="mt-1 flex items-center justify-between border-t border-white/[.06] pt-3 text-[10px] text-slate-500"><span>Health score <span className="ml-1 text-slate-300">0—100</span></span><span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-violet-400" /> Inference points</span></div>
  </section>;
}
