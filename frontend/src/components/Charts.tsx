import type { Dict } from "../api";

const W = 800, H = 180;

function scale(vals: number[], lo = Math.min(...vals), hi = Math.max(...vals)) {
  const span = hi - lo || 1;
  return (v: number) => H - 8 - ((v - lo) / span) * (H - 16);
}

/** Equity (net of all costs) with drawdown area underneath. */
export function EquityChart({ equity }: { equity: [number, number][] }) {
  if (!equity?.length) return <p className="empty">No equity data.</p>;
  const vals = equity.map((e) => e[1]);
  const y = scale(vals);
  const x = (i: number) => (i / Math.max(1, vals.length - 1)) * W;
  let peak = -Infinity;
  const dd = vals.map((v) => { peak = Math.max(peak, v); return 1 - v / peak; });
  const maxDd = Math.max(...dd, 0.0001);
  const eqPath = vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ddPath = `M0,${H} ` + dd.map((d, i) => `L${x(i).toFixed(1)},${(H - (d / maxDd) * 40).toFixed(1)}`).join("") + ` L${W},${H} Z`;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
      aria-label={`Equity from ${vals[0].toFixed(0)} to ${vals[vals.length - 1].toFixed(0)}, max drawdown ${(maxDd * 100).toFixed(2)}%`}>
      <path className="dd" d={ddPath} />
      <path className="eq" d={eqPath} />
    </svg>
  );
}

/** Close price over a segment with entry markers for simulated trades. */
export function MarketChart({ bars, start, end, trades }: { bars: number[][]; start: number; end: number; trades: Dict[] }) {
  const seg = bars.slice(start, end);
  if (!seg.length) return <p className="empty">No market data for segment.</p>;
  const closes = seg.map((b) => b[4]);
  const y = scale(closes, Math.min(...seg.map((b) => b[3])), Math.max(...seg.map((b) => b[2])));
  const x = (i: number) => (i / Math.max(1, seg.length - 1)) * W;
  const path = closes.map((c, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(c).toFixed(1)}`).join("");
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Price with simulated entries">
      <path className="px" d={path} />
      {trades.map((t) => (
        <circle key={t.trade_id} className={t.side > 0 ? "long" : "short"} r={4}
          cx={x(t.opened_t - start)} cy={y(t.entry_mid)}>
          <title>{`${t.side > 0 ? "LONG" : "SHORT"} @ ${t.entry_px.toFixed(2)} → ${t.exit_reason} net ${t.net_pnl.toFixed(2)}`}</title>
        </circle>
      ))}
    </svg>
  );
}
