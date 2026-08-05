/**
 * Decision inspector — provenance, and the rewind contrast.
 *
 * Two things happen here, and the second is the one that sells the project.
 *
 * 1. PROVENANCE. For a chosen decision, the memories that drove it, each with
 *    the similarity and rank recorded *at decision time* and its status
 *    *today*. The header badge is driven by the explanation's top-level
 *    `has_invalidated_memories` / `invalidated_count`, so a suspect decision is
 *    flagged without walking the list; each retracted or superseded row is then
 *    marked red individually.
 *
 * 2. REWIND. Two agent runs, side by side and given the most visual weight on
 *    the page: a faithful replay at the decision's own timestamp, and a re-run
 *    against today. When a supporting memory has since been withdrawn, the left
 *    panel reproduces the original action and the right one does not. That
 *    contrast — same agent, same question, different memory — is the argument.
 *    Either panel alone proves nothing, so they are always rendered together.
 */

import { useState } from "react";
import { api, type Decision, type RerunPair, type RerunResult } from "../api";
import { useAsync } from "../hooks";
import {
  Badge,
  Button,
  Dot,
  Empty,
  ErrorNote,
  Legend,
  Numeric,
  Panel,
  clock,
  shortId,
} from "./primitives";

/* ------------------------------------------------------------------------- */

function DecisionPicker({
  decisions,
  selected,
  onSelect,
}: {
  decisions: Decision[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  if (decisions.length === 0) {
    return <Empty>no decisions recorded — fire an incident</Empty>;
  }
  return (
    <ul className="flex max-h-44 flex-col gap-px overflow-y-auto p-2">
      {decisions.map((decision) => (
        <li key={decision.id}>
          <button
            type="button"
            onClick={() => onSelect(decision.id)}
            className={`rail ${
              decision.id === selected ? "rail-fork bg-strata" : "rail-absent"
            } w-full cursor-pointer rounded-xs px-2.5 py-2 text-left transition-colors duration-150 hover:bg-strata/60`}
          >
            <p className="truncate text-xs text-quartz">{decision.action}</p>
            <Numeric className="mt-0.5 block text-graphite">
              {shortId(decision.id)} · {decision.agent_id} · {clock(decision.created_at)}
            </Numeric>
          </button>
        </li>
      ))}
    </ul>
  );
}

/* ------------------------------------------------------------------------- */

/** A similarity score, drawn as a bar so ranks compare at a glance. */
function SimilarityBar({ value }: { value: number | null }) {
  if (value === null) {
    return <Numeric className="text-graphite">n/a</Numeric>;
  }
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-1 w-14 overflow-hidden rounded-xs bg-seam-lit">
        <span className="block h-full bg-live" style={{ width: `${pct}%` }} />
      </span>
      <Numeric className="text-vapor">{value.toFixed(3)}</Numeric>
    </span>
  );
}

/* ------------------------------------------------------------------------- */

function RunPanel({
  run,
  label,
  caption,
  emphasis,
}: {
  run: RerunResult;
  label: string;
  caption: string;
  emphasis: boolean;
}) {
  const changed = run.action_changed;
  return (
    <div
      className={`flex flex-col rounded-md border p-4 transition-colors duration-150 ${
        emphasis && changed
          ? "border-retract/60 bg-retract-wash"
          : "border-seam bg-void/40"
      }`}
    >
      <div className="mb-3 flex items-center justify-between gap-2">
        <Legend className={changed ? "text-retract" : "text-live"}>{label}</Legend>
        <Badge tone={changed ? "retract" : "live"}>
          <Dot tone={changed ? "retract" : "live"} />
          {changed ? "action changed" : "action unchanged"}
        </Badge>
      </div>

      <Legend dim>{caption}</Legend>
      <Numeric className="mb-3 mt-1 block text-graphite">
        evaluated at {clock(run.evaluated_at)}
      </Numeric>

      <p
        className={`text-sm leading-relaxed ${
          changed ? "text-signal" : "text-quartz"
        }`}
      >
        {run.new_action}
      </p>

      {run.new_rationale && (
        <p className="mt-2 border-t border-seam pt-2 text-xs leading-relaxed text-vapor">
          {run.new_rationale}
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-2 border-t border-seam pt-3">
        <Badge tone="retract">{run.memory_diff.only_then.length} gone since</Badge>
        <Badge tone="learned">{run.memory_diff.only_now.length} learned since</Badge>
        <Badge tone="neutral">
          {run.memory_diff.then_count} → {run.memory_diff.now_count} memories
        </Badge>
      </div>
    </div>
  );
}

function RewindContrast({ pair }: { pair: RerunPair }) {
  return (
    <div className="border-t-2 border-seam-lit bg-substrate p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Legend className="text-signal">rewind · two runs, one question</Legend>
          <Numeric className="text-graphite">reasoner {pair.reasoner}</Numeric>
        </div>
        {pair.verdict_changed ? (
          <Badge tone="retract">
            <Dot tone="retract" pulse />
            today the agent decides differently
          </Badge>
        ) : (
          <Badge tone="live">
            <Dot tone="live" />
            today the agent decides the same
          </Badge>
        )}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <RunPanel
          run={pair.faithful}
          label="faithful replay"
          caption="re-run against exactly what the branch knew at decision time"
          emphasis={false}
        />
        <RunPanel
          run={pair.today}
          label="re-run against today"
          caption="re-run against what the branch knows now"
          emphasis
        />
      </div>

      {pair.verdict_changed && (
        <p className="mt-3 text-xs leading-relaxed text-vapor">
          Same agent, same incident. The faithful replay reproduces the original
          action, which proves the replay is honest. The re-run against today
          reaches a different action — so the decision did not change because the
          agent changed, it changed because the memory did.
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------------- */

export function DecisionInspector({
  branch,
  branchName,
  refreshKey,
}: {
  /** Branch id — used for querying. */
  branch: string | null;
  /** Branch name — used for display. */
  branchName: string | null;
  refreshKey: number;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [pair, setPair] = useState<RerunPair | null>(null);
  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);

  const decisions = useAsync(() => api.decisions(branch ?? undefined), [branch, refreshKey]);
  const list = decisions.data?.decisions ?? [];
  const activeId = selected ?? list[0]?.id ?? null;

  const explanation = useAsync(
    () => (activeId ? api.explain(activeId) : Promise.resolve(null)),
    [activeId, refreshKey],
  );
  const explained = explanation.data;

  const runRewind = async () => {
    if (!activeId) return;
    setRerunning(true);
    setRerunError(null);
    try {
      setPair(await api.rerun(activeId));
    } catch (error) {
      setRerunError(error instanceof Error ? error.message : String(error));
    } finally {
      setRerunning(false);
    }
  };

  const select = (id: string) => {
    setSelected(id);
    setPair(null); // a rewind belongs to one decision; never carry it across
    setRerunError(null);
  };

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <Panel
        title="decisions"
        meta={<Numeric className="text-vapor">{branchName ?? "all branches"}</Numeric>}
      >
        {decisions.error ? (
          <ErrorNote>{decisions.error}</ErrorNote>
        ) : (
          <DecisionPicker decisions={list} selected={activeId} onSelect={select} />
        )}
      </Panel>

      {!activeId ? null : explanation.error ? (
        <Panel title="provenance">
          <ErrorNote>{explanation.error}</ErrorNote>
        </Panel>
      ) : !explained ? (
        <Panel title="provenance">
          <Empty>reading provenance…</Empty>
        </Panel>
      ) : (
        <Panel
          title="provenance"
          meta={
            /* Driven by the top-level flag — no list walk needed to know a
               decision is suspect. */
            explained.has_invalidated_memories ? (
              <Badge tone="retract">
                <Dot tone="retract" pulse />
                rested on {explained.invalidated_count} withdrawn{" "}
                {explained.invalidated_count === 1 ? "memory" : "memories"}
              </Badge>
            ) : (
              <Badge tone="live">
                <Dot tone="live" />
                all supporting memory still stands
              </Badge>
            )
          }
          actions={
            <Button tone="fork" onClick={runRewind} disabled={rerunning}>
              {rerunning ? "rewinding…" : "rewind"}
            </Button>
          }
          bodyClassName="flex flex-col min-h-0"
        >
          <div className="border-b border-seam px-4 py-3">
            <Legend dim>action taken</Legend>
            <p className="mt-1 text-sm leading-relaxed text-signal">
              {explained.decision.action}
            </p>
            {explained.decision.rationale && (
              <p className="mt-2 text-xs leading-relaxed text-vapor">
                {explained.decision.rationale}
              </p>
            )}
            <Numeric className="mt-2 block text-graphite">
              {explained.branch_name} · {explained.decision.agent_id} ·{" "}
              {clock(explained.decision.created_at)}
            </Numeric>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            <ul className="flex flex-col gap-px">
              {explained.memories.map((memory) => (
                <li
                  key={memory.memory_id}
                  className={`rail ${
                    memory.invalidated ? "rail-retract" : "rail-active"
                  } rounded-xs px-3 py-2.5 hover:bg-strata/50 ${
                    memory.invalidated ? "withdrawn" : ""
                  }`}
                >
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    <Numeric className="text-vapor">#{memory.rank}</Numeric>
                    <SimilarityBar value={memory.similarity} />
                    <Legend dim>{memory.kind}</Legend>
                    {memory.invalidated && (
                      <Badge tone="retract">{memory.status_now}</Badge>
                    )}
                  </div>
                  <p
                    className={`text-xs leading-relaxed text-quartz ${
                      memory.invalidated ? "withdrawn-text" : ""
                    }`}
                  >
                    {memory.content}
                  </p>
                  <Numeric className="mt-1 block text-graphite">
                    {memory.source ?? "unknown source"}
                    {memory.retracted_at && ` · retracted ${clock(memory.retracted_at)}`}
                    {memory.superseded_at && ` · superseded ${clock(memory.superseded_at)}`}
                  </Numeric>
                </li>
              ))}
            </ul>
          </div>

          {rerunError && <ErrorNote>{rerunError}</ErrorNote>}
          {pair && <RewindContrast pair={pair} />}
        </Panel>
      )}
    </div>
  );
}
