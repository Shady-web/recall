/**
 * The header: wordmark plus the cluster facts the whole demo rests on.
 *
 * Everything here is read from `/api/health` and rendered verbatim — the
 * CockroachDB build string, the actor identity, which embedding and reasoning
 * providers are actually wired up, and whether the live feed is a real
 * changefeed or the polling fallback. Nothing is asserted that was not read.
 *
 * The offline badge is deliberately loud. Running against the fake embedding
 * provider and the rule reasoner is the right way to iterate, and it would be
 * dishonest for that mode to look identical to a real Bedrock run.
 */

import type { Health } from "../api";
import { Badge, Dot, Legend, Numeric } from "./primitives";

/**
 * Pull the version number out of the cluster's build string.
 *
 * `SELECT version()` returns something like
 * `CockroachDB CCL v26.2.4 (aarch64-unknown-linux-gnu, built …)`. Showing the
 * leading words next to a label that already says COCKROACHDB just prints the
 * product name twice; the version is the part that carries information. The
 * full build string stays available in the element's tooltip.
 */
function clusterVersion(raw: string | null | undefined): string {
  if (!raw) return "connected";
  return raw.split(/\s+/).find((token) => /^v\d/.test(token)) ?? raw.slice(0, 24);
}

export function StatusBar({
  health,
  feedMode,
  feedConnected,
  eventCount,
}: {
  health: Health | null;
  feedMode: string;
  feedConnected: boolean;
  eventCount: number;
}) {
  const reachable = health?.cluster_reachable ?? false;
  const pushing = feedConnected && feedMode === "changefeed";

  return (
    <header className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-2 border-b border-seam bg-substrate px-4 py-2.5">
      <div className="flex items-baseline gap-2.5">
        <span className="font-mono text-sm font-medium tracking-[0.18em] text-signal">
          RECALL
        </span>
        <Legend dim>agent memory console</Legend>
      </div>

      <span className="h-4 w-px bg-seam-lit" />

      {/* Cluster */}
      <span className="flex items-center gap-2" title={health?.cluster_version ?? ""}>
        <Dot tone={reachable ? "live" : "retract"} pulse={reachable} />
        <Legend dim>cockroachdb</Legend>
        <Numeric className={reachable ? "text-live" : "text-retract"}>
          {reachable ? clusterVersion(health?.cluster_version) : "unreachable"}
        </Numeric>
      </span>

      {/* Live feed — reports what it actually is */}
      <span
        className="flex items-center gap-2"
        title={
          pushing
            ? "CockroachDB changefeed streaming row changes over SSE"
            : "Changefeed unavailable; the backend is polling instead"
        }
      >
        <Dot tone={pushing ? "live" : feedConnected ? "learned" : "retract"} />
        <Legend dim>feed</Legend>
        <Numeric className={pushing ? "text-live" : "text-learned"}>
          {feedConnected ? feedMode : "disconnected"}
        </Numeric>
        {eventCount > 0 && <Numeric className="text-graphite">{eventCount} events</Numeric>}
      </span>

      <span className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
        {health?.offline_mode && (
          <Badge tone="learned">
            offline mode · fake embeddings · rule reasoner
          </Badge>
        )}
        {health?.read_only && <Badge tone="retract">read-only kernel</Badge>}

        <span className="flex items-center gap-2">
          <Legend dim>embed</Legend>
          <Numeric className="text-vapor">
            {health?.embedding_provider ?? "—"}
          </Numeric>
        </span>

        <span className="flex items-center gap-2">
          <Legend dim>reason</Legend>
          <Numeric className="text-vapor">{health?.reasoning_provider ?? "—"}</Numeric>
        </span>

        <span className="flex items-center gap-2">
          <Legend dim>actor</Legend>
          <Numeric className="text-vapor">{health?.actor ?? "—"}</Numeric>
        </span>
      </span>
    </header>
  );
}
