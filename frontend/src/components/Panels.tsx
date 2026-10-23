import { Fragment } from "react";
import { fmt, type Dict } from "../api";

export const statusClass = (s: string) =>
  ({ QUALIFIED: "s-good", RETAINED: "s-good", RETAIN: "s-good", APPROVE: "s-good", COMPLETED: "s-good",
     RESIZE: "s-info", DEFER: "s-warn", INSUFFICIENT_EVIDENCE: "s-warn", NO_TRADE: "s-muted",
     REJECT: "s-bad", REJECTED: "s-bad", FAILED_RISK: "s-bad", ABORTED_INTEGRITY: "s-bad" } as Dict)[s] ?? "s-info";

export const Pill = ({ s }: { s: string }) => <span className={`pill ${statusClass(s)}`}>{s}</span>;

export function KV({ obj, keys }: { obj: Dict | null | undefined; keys?: string[] }) {
  if (!obj) return <p className="empty">Unavailable.</p>;
  return (
    <dl className="kv">
      {(keys ?? Object.keys(obj)).map((k) => (
        <Fragment key={k}><dt>{k}</dt><dd>{typeof obj[k] === "object" && obj[k] !== null ? JSON.stringify(obj[k]) : fmt(obj[k])}</dd></Fragment>
      ))}
    </dl>
  );
}

export function FoldScorecard({ candidates, folds }: { candidates: Dict[]; folds: Dict | null }) {
  return (
    <div className="panel">
      <h2>Fold-by-fold OOS scorecards</h2>
      {folds && (
        <p className="mono s-muted">
          {folds.folds.map((f: Dict) => `F${f.k}: IS[${f.is_start},${f.is_end}) OOS[${f.oos_start},${f.oos_end})`).join("  ·  ")}
          {"  ·  "}holdout[{folds.holdout.join(",")}) embargo {folds.embargo}
        </p>
      )}
      <div className="scroll">
        <table>
          <thead><tr><th>fold</th><th>chromosome</th><th>status</th><th>judge</th><th>trades</th><th>net</th><th>gross</th><th>costs</th><th>funding</th><th>max DD</th><th>exposure</th><th>hold</th></tr></thead>
          <tbody>
            {candidates.map((c) => {
              const s = c.oos.scorecard;
              return (
                <tr key={c.fold + c.chromosome.id}>
                  <td>{c.fold}</td><td>{c.chromosome.id}</td><td><Pill s={c.status} /></td><td><Pill s={c.judge.verdict} /></td>
                  <td>{s.n_trades}</td>
                  <td className={s.net_pnl >= 0 ? "s-good" : "s-bad"}>{fmt(s.net_pnl, 2)}</td>
                  <td>{fmt(s.gross_pnl, 2)}</td><td>{fmt(s.cost_total, 2)}</td><td>{fmt(s.funding_total, 2)}</td>
                  <td>{fmt(s.max_drawdown * 100, 2)}%</td><td>{fmt(s.max_exposure, 2)}x</td><td>{fmt(s.avg_bars_held, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function GenomeLineage({ c, all }: { c: Dict; all: Dict[] }) {
  const parents = all.filter((x) => c.parents.includes(x.id));
  return (
    <div className="panel">
      <h2>Genome & lineage</h2>
      <p className="mono">{c.id} · gen {c.generation} · {c.operator}</p>
      <p className="mono s-muted">parents: {c.parents.length ? c.parents.join(", ") : "none (random init)"}
        {parents.length ? ` (fitness ${parents.map((p) => fmt(p.fitness, 2)).join(", ")})` : ""}</p>
      {c.changed?.length > 0 && <p className="mono s-warn">mutations: {c.changed.join(" | ")}</p>}
      {c.repairs?.length > 0 && <p className="mono s-info">repairs: {c.repairs.join(" | ")}</p>}
      <div className="grid2">
        {Object.entries(c.genes as Dict).map(([fam, g]) => (
          <div key={fam}><h3 className="mono s-muted" style={{ fontSize: 11 }}>{fam}</h3><KV obj={g as Dict} /></div>
        ))}
      </div>
    </div>
  );
}

export function Ledger({ trades }: { trades: Dict[] }) {
  return (
    <div className="panel">
      <h2>Simulated fills & costs (paper)</h2>
      {!trades.length ? <p className="empty">No trades in this segment.</p> : (
        <div className="scroll">
          <table>
            <thead><tr><th>trade</th><th>side</th><th>qty</th><th>entry</th><th>exit</th><th>reason</th><th>gross</th><th>spread</th><th>slip</th><th>fees</th><th>funding</th><th>net</th><th>stop ok</th><th>disc.</th></tr></thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.trade_id}>
                  <td title={t.trade_id}>{t.trade_id.split(":").slice(-2).join(":")}</td>
                  <td className={t.side > 0 ? "s-good" : "s-bad"}>{t.side > 0 ? "LONG" : "SHORT"}</td>
                  <td>{fmt(t.qty, 4)}</td><td>{fmt(t.entry_px, 2)}</td><td>{fmt(t.exit_px, 2)}</td><td>{t.exit_reason}</td>
                  <td>{fmt(t.gross_mid_pnl, 2)}</td><td>{fmt(t.spread_cost, 2)}</td><td>{fmt(t.slippage_cost, 2)}</td>
                  <td>{fmt(t.fees, 2)}</td><td>{fmt(t.funding, 2)}</td>
                  <td className={t.net_pnl >= 0 ? "s-good" : "s-bad"}>{fmt(t.net_pnl, 2)}</td>
                  <td>{fmt(t.stop_compliant)}</td><td>{t.adoption}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function DecisionView({ d, narration }: { d: Dict; narration?: Dict }) {
  const p = d.proposal, r = d.risk, disc = d.discussion;
  return (
    <div className="panel">
      <h2>Decision @ bar {d.t} · {new Date(d.ts * 1000).toISOString().slice(0, 16)}Z</h2>
      {disc ? (
        <>
          <p className="mono">Discussion score {fmt(disc.score, 3)} → {disc.recommendation}</p>
          {disc.turns.map((t: Dict) => (
            <div key={t.role} className={`turn ${t.role}`}>
              <strong>{t.role.toUpperCase()}</strong> <span className="s-muted mono">uncertainty {fmt(t.uncertainty, 2)}</span>
              <ul>{t.claims.map((c: Dict, i: number) => <li key={i}>{c.text} <span className="ev">{c.evidence_id}</span></li>)}
                {!t.claims.length && <li className="empty">no evidence-backed claim</li>}</ul>
              <div className="s-muted" style={{ fontSize: 12 }}>counter: {t.counterarguments.join("; ")} · invalidation: {t.invalidation}</div>
            </div>
          ))}
        </>
      ) : <p className="empty">No discussion (insufficient evidence).</p>}
      <h2 style={{ marginTop: 12 }}>Trader proposal</h2>
      <p className="mono">{p.side.toUpperCase()} qty {fmt(p.qty, 4)} · stop {fmt(p.stop, 2)} · target {fmt(p.target, 2)} · discussion <Pill s={p.discussion_adoption} /></p>
      <p className="s-muted" style={{ fontSize: 12 }}>{p.rationale}</p>
      <h2 style={{ marginTop: 12 }}>Risk Control (authoritative) <Pill s={r.outcome} /></h2>
      <p className="mono">requested {fmt(r.requested_qty, 4)} → approved {fmt(r.approved_qty, 4)} · {r.risk_version}</p>
      {r.checks.map((c: Dict) => (
        <div key={c.name} className="check">
          <span className={c.passed ? "s-good" : c.detail === "resized" ? "s-info" : "s-bad"}>{c.passed ? "PASS" : c.detail === "resized" ? "CAP " : "FAIL"}</span>
          <span>{c.name}</span><span className="s-muted">{fmt(c.value)} / {fmt(c.limit)}</span>
        </div>
      ))}
      <p className="mono s-muted">{r.reasons.join("; ")}</p>
      {narration && (
        <>
          <h2 style={{ marginTop: 12 }}>Narration (non-authoritative · {narration.provider})</h2>
          {narration.error ? <p className="err">Provider error: {narration.error}</p> : <p style={{ fontSize: 13 }}>{narration.text}</p>}
          {narration.unsupported_ids?.length > 0 && <p className="err">Unsupported evidence cited: {narration.unsupported_ids.join(", ")}</p>}
        </>
      )}
    </div>
  );
}
