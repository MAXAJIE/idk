import { useEffect, useMemo, useState } from "react";
import { api, fmt, streamEvents, type Dict } from "../api";
import { EquityChart, MarketChart } from "../components/Charts";
import { DecisionView, FoldScorecard, GenomeLineage, KV, Ledger, Pill } from "../components/Panels";

/**
 * Read-only spectator. Deliberately contains no prompt box, trade button,
 * parameter editor or run control: the backend exposes none either.
 */
export default function Spectator() {
  const [exps, setExps] = useState<Dict[]>([]);
  const [eid, setEid] = useState<string>("");
  const [exp, setExp] = useState<Dict | null>(null);
  const [bars, setBars] = useState<number[][]>([]);
  const [cands, setCands] = useState<Dict[]>([]);
  const [chroms, setChroms] = useState<Dict[]>([]);
  const [final, setFinal] = useState<{ final: Dict | null; holdout: Dict[] } | null>(null);
  const [events, setEvents] = useState<Dict[]>([]);
  const [sel, setSel] = useState<string>("");
  const [seg, setSeg] = useState<"oos" | "is">("oos");
  const [dec, setDec] = useState(0);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    api.experiments().then((x) => { setExps(x); if (x[0]) setEid(x[0].id); }).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!eid) return;
    setEvents([]); setSel("");
    const load = () => Promise.all([api.experiment(eid), api.candidates(eid), api.chromosomes(eid), api.final(eid)])
      .then(([e, c, ch, f]) => { setExp(e); setCands(c); setChroms(ch); setFinal(f); })
      .catch((e) => setErr(String(e)));
    load();
    api.bars(eid).then((b) => setBars(b.bars)).catch(() => setBars([]));
    const stop = streamEvents(eid, (e) => {
      setEvents((xs) => [e, ...xs].slice(0, 400));
      if (["judge_verdict", "final_ranking", "state"].includes(e.type)) load();
    });
    return stop;
  }, [eid]);

  const cand = useMemo(() => cands.find((c) => `${c.fold}:${c.chromosome.id}` === sel) ?? cands[0], [cands, sel]);
  const ranking: Dict[] = final?.final?.ranking ?? [];
  const holdout = final?.holdout?.[0];
  const segData = cand?.[seg];
  const decisions: Dict[] = cand?.decisions?.filter((d: Dict) => d.proposal.side !== "no_trade" || d.proposal.discussion_adoption !== "NOT_EVALUATED") ?? [];
  const decision = decisions[Math.min(dec, decisions.length - 1)];

  return (
    <>
      <div className="banner">PAPER / BACKTEST RESEARCH ONLY · NO LIVE ORDERS · SIMULATED FILLS AND ESTIMATED COSTS · NOT FINANCIAL ADVICE</div>
      <header className="top">
        <h1>CGTRS <span>Spectator</span></h1>
        <select aria-label="Experiment" value={eid} onChange={(e) => setEid(e.target.value)}>
          {exps.map((x) => <option key={x.id} value={x.id}>{x.name} · {x.id}</option>)}
        </select>
        {exp && <Pill s={exp.state} />}
        {exp && <span className={`mono ${exp.chain_ok ? "s-good" : "s-bad"}`}>event chain {exp.chain_ok ? "verified" : "BROKEN"}</span>}
        <span className="spacer" />
        {err && <span className="err">{err}</span>}
        {!exps.length && !err && <span className="empty">No experiments yet. Run one from the terminal.</span>}
      </header>

      <div className="layout">
        <aside>
          <div className="panel">
            <h2>Data provenance</h2>
            <KV obj={exp?.dataset} keys={["source", "exchange", "symbol", "timeframe_minutes", "n_bars", "checksum"]} />
            {(exp?.dataset?.quality_errors?.length ?? 0) > 0 && <p className="err">{exp!.dataset.quality_errors.join("; ")}</p>}
            <h2 style={{ marginTop: 12 }}>Versions</h2>
            <KV obj={exp && { risk: exp.config.risk.version, features: exp.config.feature_version, schema: exp.config.schema_version, ga_seed: exp.config.ga.seed, llm: exp.config.llm.provider }} />
          </div>
          <div className="panel">
            <h2>Final WFA ranking</h2>
            {!ranking.length && <p className="empty">No retained chromosomes (yet).</p>}
            {ranking.map((r) => (
              <div key={r.id} className="row" onClick={() => setSel(`${r.fold}:${r.id}`)}>
                <div className="id">{r.id} <Pill s={r.status} /></div>
                <div className="meta">fold {r.fold} · {r.windows} OOS windows · consistency {fmt(r.consistency, 2)} · Judge-adj net {fmt(r.rank_net, 2)}</div>
                {r.holdout && <div className="meta">holdout net {fmt(r.holdout.net_pnl, 2)} · DD {fmt(r.holdout.max_drawdown * 100, 2)}% · {r.holdout.n_trades} trades</div>}
              </div>
            ))}
          </div>
          <div className="panel">
            <h2>Frozen candidates</h2>
            {cands.map((c) => {
              const k = `${c.fold}:${c.chromosome.id}`;
              return (
                <div key={k} className={`row ${cand && k === `${cand.fold}:${cand.chromosome.id}` ? "sel" : ""}`} onClick={() => { setSel(k); setDec(0); }}>
                  <div className="id">F{c.fold} {c.chromosome.id} <Pill s={c.status} /></div>
                  <div className="meta">{c.chromosome.genes.topology.entry_mode} · OOS net {fmt(c.oos.scorecard.net_pnl, 2)} · robustness {fmt(c.robustness.retention, 2)}</div>
                </div>
              );
            })}
          </div>
        </aside>

        <main>
          <FoldScorecard candidates={cands} folds={exp?.folds ?? null} />
          {cand && (
            <>
              <div className="panel">
                <h2>
                  {cand.chromosome.id} · segment{" "}
                  <select aria-label="Segment" value={seg} onChange={(e) => setSeg(e.target.value as "oos" | "is")}>
                    <option value="oos">out-of-sample</option><option value="is">in-sample</option>
                  </select>
                </h2>
                <MarketChart bars={bars} start={segData.start} end={segData.end} trades={segData.trades} />
                <EquityChart equity={segData.equity} />
                <div className="grid2" style={{ marginTop: 10 }}>
                  <KV obj={segData.scorecard} keys={["net_pnl", "gross_pnl", "cost_total", "funding_total", "cost_share_of_gross", "max_drawdown", "sharpe_bar", "sortino_bar"]} />
                  <KV obj={segData.scorecard} keys={["n_trades", "win_rate", "stop_compliance", "avg_bars_held", "avg_exposure", "max_exposure", "time_in_market", "worst_trade"]} />
                </div>
                <p className="mono s-muted">risk outcomes: {JSON.stringify(segData.risk_counts)}{segData.failed_risk ? ` · FAILED_RISK at bar ${segData.failed_risk.t}` : ""}</p>
              </div>
              <div className="grid2">
                <div className="panel">
                  <h2>Qualification gates (OOS)</h2>
                  {cand.oos_gates.map((g: Dict) => (
                    <div key={g.gate} className="check"><span className={g.passed ? "s-good" : "s-bad"}>{g.passed ? "PASS" : "FAIL"}</span><span>{g.gate}</span><span className="s-muted">{fmt(g.value)} / {fmt(g.limit)}</span></div>
                  ))}
                </div>
                <div className="panel">
                  <h2>Judge verdict <Pill s={cand.judge.verdict} /></h2>
                  <p style={{ fontSize: 13 }}>{cand.judge.rationale}</p>
                  <KV obj={cand.judge} keys={["adjusted_net_pnl", "adjusted_max_drawdown", "uncertainty", "nullified_trade_ids", "flags", "risk_concerns"]} />
                </div>
              </div>
              <Ledger trades={segData.trades} />
              <GenomeLineage c={cand.chromosome} all={chroms} />
              {decisions.length > 0 && (
                <>
                  <div className="panel">
                    <h2>Recorded OOS decisions</h2>
                    <select aria-label="Decision" value={dec} onChange={(e) => setDec(Number(e.target.value))}>
                      {decisions.map((d, i) => <option key={i} value={i}>bar {d.t} · {d.proposal.side} · {d.risk.outcome}</option>)}
                    </select>
                  </div>
                  {decision && <DecisionView d={decision} />}
                </>
              )}
            </>
          )}
          {holdout && holdout.narration?.length > 0 && (
            <div className="panel">
              <h2>Holdout narration (top qualified chromosome)</h2>
              {holdout.narration.map((n: Dict) => {
                const d = holdout.decisions.find((x: Dict) => x.proposal.decision_id === n.decision_id);
                return d ? <DecisionView key={n.decision_id} d={d} narration={n} /> : null;
              })}
            </div>
          )}
        </main>

        <aside>
          <div className="panel">
            <h2>Live event stream</h2>
            <div className="events">
              {!events.length && <p className="empty">Waiting for events…</p>}
              {events.map((e) => (
                <div key={e.seq}><span className="t">#{e.seq} {e.type}</span> {JSON.stringify(e.payload).slice(0, 160)}</div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}
