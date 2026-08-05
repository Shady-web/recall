/**
 * Timeline scrubber — drag across time, watch what the branch knew.
 *
 * SMOOTHNESS. Every position implies a replay query, so the control is split in
 * two: the playhead and clock follow the drag with zero delay, while the fetch
 * runs off a debounced value. While a fetch is in flight the previous result
 * stays on screen at reduced opacity instead of blanking, so dragging never
 * flickers or empties. The scrub head is a thin vertical bar rather than a knob
 * because it is a playhead over a time axis.
 *
 * THEN VS NOW. The list is the union of two live reads: `replay(branch, t)` and
 * `replay(branch, now)`. Rows are marked by which set they belong to —
 *   - in both        → normal, live rail
 *   - only at `t`    → struck through and dimmed (it was available then, and has
 *                      since been retracted or superseded)
 *   - only now       → highlighted on the learned rail (the branch had not
 *                      learned it yet at this instant)
 * That difference is the whole point of a replayable memory, so it is rendered
 * rather than described.
 *
 * THE FORENSIC LIMIT. Physical replay (`AS OF SYSTEM TIME`) cannot reach past
 * the cluster's MVCC garbage-collection window. Rather than letting that fail
 * at request time, the control is disabled whenever the scrubbed instant falls
 * outside the window the server reported, and says why. A limit you can see is
 * a feature; a limit that surfaces as a broken request is a bug.
 */

import { useEffect, useMemo, useState } from "react";
import { api, type Memory, type Timeline } from "../api";
import { useAsync, useDebounced, useLastKnown } from "../hooks";
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  Legend,
  Numeric,
  Panel,
  clock,
  timeOnly,
} from "./primitives";

type Availability = "both" | "only-then" | "only-now";

interface Row {
  memory: Memory;
  availability: Availability;
}

function buildRows(then: Memory[], now: Memory[]): Row[] {
  const thenIds = new Set(then.map((m) => m.id));
  const nowIds = new Set(now.map((m) => m.id));
  const merged = new Map<string, Row>();

  for (const memory of then) {
    merged.set(memory.id, {
      memory,
      availability: nowIds.has(memory.id) ? "both" : "only-then",
    });
  }
  for (const memory of now) {
    if (!thenIds.has(memory.id)) merged.set(memory.id, { memory, availability: "only-now" });
  }

  return [...merged.values()].sort(
    (a, b) => Date.parse(a.memory.created_at) - Date.parse(b.memory.created_at),
  );
}

const RAIL: Record<Availability, string> = {
  both: "rail-active",
  "only-then": "rail-retract",
  "only-now": "rail-learned",
};

function MemoryRow({ row }: { row: Row }) {
  const gone = row.availability === "only-then";
  const learned = row.availability === "only-now";
  return (
    <li
      className={`rail ${RAIL[row.availability]} rise rounded-xs px-3 py-2.5 transition-colors duration-150 hover:bg-strata/50 ${
        gone ? "withdrawn" : ""
      } ${learned ? "bg-learned-wash" : ""}`}
    >
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <Legend dim>{row.memory.kind}</Legend>
        {gone && <Badge tone="retract">gone now · {row.memory.status}</Badge>}
        {learned && <Badge tone="learned">learned since</Badge>}
        <Numeric className="ml-auto text-graphite">{clock(row.memory.created_at)}</Numeric>
      </div>
      <p className={`text-xs leading-relaxed text-quartz ${gone ? "withdrawn-text" : ""}`}>
        {row.memory.content}
      </p>
      {row.memory.source && (
        <Numeric className="mt-1 block text-graphite">{row.memory.source}</Numeric>
      )}
    </li>
  );
}

export function TimelineScrubber({
  branch,
  branchName,
  timeline,
  refreshKey,
}: {
  /** Branch id — used for querying, because it is stable and slash-free. */
  branch: string;
  /** Branch name — used for display; an operator reads `main`, not a UUID. */
  branchName: string;
  timeline: Timeline;
  refreshKey: number;
}) {
  const earliest = Date.parse(timeline.earliest);
  const latest = Date.parse(timeline.latest);
  const span = Math.max(latest - earliest, 1000);

  // Start at "now" so the console opens on current reality.
  const [position, setPosition] = useState(latest);
  const [physical, setPhysical] = useState(false);

  useEffect(() => {
    setPosition(Date.parse(timeline.latest));
  }, [timeline.latest, branch]);

  // Immediate value drives the playhead; settled value drives the query.
  const settled = useDebounced(position, 90);
  const settledIso = useMemo(() => new Date(settled).toISOString(), [settled]);
  const nowIso = timeline.latest;

  const gcEarliest = Date.parse(timeline.replay_window.earliest);
  const withinReplayWindow = settled >= gcEarliest && settled <= latest;

  // Physical replay is refused outside the window, so drop back to logical
  // rather than issuing a request the server will reject.
  const mode = physical && withinReplayWindow ? "physical" : "logical";

  const thenQuery = useAsync(
    () => api.replay(branch, settledIso, mode),
    [branch, settledIso, mode, refreshKey],
  );
  const nowQuery = useAsync(
    () => api.replay(branch, nowIso, "logical"),
    [branch, nowIso, refreshKey],
  );

  const thenMemories = useLastKnown(thenQuery.data?.memories ?? null) ?? [];
  const nowMemories = useLastKnown(nowQuery.data?.memories ?? null) ?? [];
  const rows = useMemo(
    () => buildRows(thenMemories, nowMemories),
    [thenMemories, nowMemories],
  );

  const goneCount = rows.filter((r) => r.availability === "only-then").length;
  const learnedCount = rows.filter((r) => r.availability === "only-now").length;
  const inFlight = thenQuery.loading || nowQuery.loading;
  const atNow = latest - settled < 1500;

  const forkOffset =
    timeline.fork_point_ts !== null
      ? ((Date.parse(timeline.fork_point_ts) - earliest) / span) * 100
      : null;

  return (
    <Panel
      title="timeline"
      meta={<Numeric className="text-vapor">{branchName}</Numeric>}
      actions={
        <>
          <Button
            tone={physical ? "live" : "neutral"}
            disabled={!withinReplayWindow}
            onClick={() => setPhysical((p) => !p)}
            title={
              withinReplayWindow
                ? "Read true historical bytes with AS OF SYSTEM TIME"
                : `Outside the MVCC garbage-collection window ` +
                  `(gc.ttlseconds=${timeline.replay_window.gc_ttl_seconds}). ` +
                  `Physical replay can only reach back to ` +
                  `${clock(timeline.replay_window.earliest)}. Logical replay still works.`
            }
          >
            forensic replay {physical && withinReplayWindow ? "on" : "off"}
          </Button>
          <Button onClick={() => setPosition(latest)} disabled={atNow}>
            jump to now
          </Button>
        </>
      }
      bodyClassName="flex flex-col min-h-0"
    >
      {/* ---- the control ------------------------------------------------ */}
      <div className="shrink-0 border-b border-seam px-4 pb-3 pt-3">
        <div className="mb-1 flex items-baseline justify-between gap-4">
          <div className="flex items-baseline gap-3">
            <Legend dim>branch knew, as of</Legend>
            <span className="font-mono text-lg tabular-nums tracking-tight text-signal">
              {timeOnly(new Date(position).toISOString())}
            </span>
            {atNow && <Badge tone="live">now</Badge>}
          </div>
          <Numeric className="text-graphite">
            {clock(new Date(position).toISOString())}
          </Numeric>
        </div>

        <div className="relative">
          <input
            type="range"
            className="scrub"
            min={earliest}
            max={latest}
            step={250}
            value={position}
            aria-label="Reconstruct branch state at this instant"
            onChange={(event) => setPosition(Number(event.target.value))}
          />
          {/* Fork point marker — where this branch began to diverge. */}
          {forkOffset !== null && forkOffset >= 0 && forkOffset <= 100 && (
            <span
              className="pointer-events-none absolute top-1/2 h-3 w-px -translate-y-1/2 bg-fork"
              style={{ left: `${forkOffset}%` }}
              title="fork point"
            />
          )}
          {/* Edge of physical replay — the forensic horizon. */}
          {gcEarliest > earliest && gcEarliest < latest && (
            <span
              className="pointer-events-none absolute top-1/2 h-3 w-px -translate-y-1/2 bg-learned/60"
              style={{ left: `${((gcEarliest - earliest) / span) * 100}%` }}
              title="edge of the physical replay window"
            />
          )}
        </div>

        <div className="flex items-center justify-between">
          <Numeric className="text-graphite">{clock(timeline.earliest)}</Numeric>
          <div className="flex items-center gap-3">
            <Legend dim>
              mode {mode}
              {physical && !withinReplayWindow ? " (forensic unavailable here)" : ""}
            </Legend>
          </div>
          <Numeric className="text-graphite">{clock(timeline.latest)}</Numeric>
        </div>
      </div>

      {/* ---- the diff legend -------------------------------------------- */}
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-seam px-4 py-2">
        <Badge tone="live">{rows.length - goneCount - learnedCount} still known</Badge>
        <Badge tone="retract">{goneCount} gone since</Badge>
        <Badge tone="learned">{learnedCount} learned since</Badge>
        {inFlight && <Legend dim className="ml-auto">reading…</Legend>}
      </div>

      {/* ---- the memories ----------------------------------------------- */}
      {thenQuery.error ? (
        <ErrorNote>{thenQuery.error}</ErrorNote>
      ) : rows.length === 0 ? (
        <Empty>
          {inFlight ? "reading cluster…" : "this branch knew nothing at that instant"}
        </Empty>
      ) : (
        <ul
          className={`flex min-h-0 flex-1 flex-col gap-px overflow-y-auto p-2 ${
            inFlight ? "inflight" : ""
          }`}
        >
          {rows.map((row) => (
            <MemoryRow key={row.memory.id} row={row} />
          ))}
        </ul>
      )}
    </Panel>
  );
}
