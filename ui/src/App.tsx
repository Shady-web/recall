/**
 * Recall console.
 *
 * Layout: a persistent branch tree on the left (the lineage is context for
 * everything else), and one of two views on the right — the timeline scrubber
 * or the decision inspector. Three surfaces, all reading live, all built on the
 * Cold Forensics tokens.
 *
 * Live updates are wired the blunt, reliable way: any change event from the
 * backend bumps `refreshKey`, and every query in the tree depends on it, so a
 * memory the agent writes appears without a reload. Refetching on a push beats
 * patching client-side state, because the refetch is what guarantees the screen
 * still matches the cluster rather than matching our idea of it.
 */

import { useEffect, useState } from "react";
import { api, type Branch } from "./api";
import { useAsync, useLiveFeed } from "./hooks";
import { BranchTree } from "./components/BranchTree";
import { DecisionInspector } from "./components/DecisionInspector";
import { StatusBar } from "./components/StatusBar";
import { TimelineScrubber } from "./components/TimelineScrubber";
import { Button, Empty, ErrorNote, Legend, Numeric, Panel } from "./components/primitives";

type View = "timeline" | "decision";

export function App() {
  const [view, setView] = useState<View>("timeline");
  const [selected, setSelected] = useState<Branch | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const feed = useLiveFeed();
  const health = useAsync(() => api.health(), [refreshKey]);
  const branches = useAsync(() => api.branches(), [refreshKey]);

  // Any cluster change refetches everything currently on screen.
  useEffect(() => {
    if (feed.eventCount > 0) setRefreshKey((key) => key + 1);
  }, [feed.eventCount]);

  const branchList = branches.data?.branches ?? [];

  // Default to the newest branch that actually holds a decision — that is where
  // the story is — falling back to main.
  useEffect(() => {
    if (selected !== null || branchList.length === 0) return;
    const withDecision = [...branchList]
      .reverse()
      .find((branch) => branch.decision_count > 0);
    setSelected(withDecision ?? branchList.find((b) => b.name === "main") ?? branchList[0]);
  }, [branchList, selected]);

  // Keep the selected branch's counts fresh as events arrive.
  const current = selected
    ? (branchList.find((branch) => branch.id === selected.id) ?? selected)
    : null;

  const timeline = useAsync(
    () => (current ? api.timeline(current.id) : Promise.resolve(null)),
    [current?.id, refreshKey],
  );

  return (
    <div className="flex h-screen flex-col bg-void">
      <StatusBar
        health={health.data}
        feedMode={feed.mode}
        feedConnected={feed.connected}
        eventCount={feed.eventCount}
      />

      <main className="flex min-h-0 flex-1 gap-3 p-3">
        {/* ---- branch tree: always visible, it is the context ---------- */}
        <aside className="flex w-72 shrink-0 flex-col">
          <Panel
            title="branches"
            meta={
              <Numeric className="text-vapor">{branchList.length} total</Numeric>
            }
            className="min-h-0 flex-1"
            bodyClassName="min-h-0 overflow-hidden"
          >
            {branches.error ? (
              <ErrorNote>{branches.error}</ErrorNote>
            ) : (
              <BranchTree
                branches={branchList}
                selected={current?.id ?? null}
                onSelect={setSelected}
              />
            )}
          </Panel>
        </aside>

        {/* ---- the working surface ------------------------------------ */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
          <nav className="flex shrink-0 items-center gap-2">
            <Button
              tone={view === "timeline" ? "live" : "neutral"}
              onClick={() => setView("timeline")}
            >
              timeline scrubber
            </Button>
            <Button
              tone={view === "decision" ? "fork" : "neutral"}
              onClick={() => setView("decision")}
            >
              decision inspector
            </Button>
            {current && (
              <span className="ml-auto flex items-center gap-3">
                <Legend dim>selected</Legend>
                <Numeric className="text-quartz">{current.name}</Numeric>
              </span>
            )}
          </nav>

          <div className="flex min-h-0 flex-1 flex-col">
            {!current ? (
              <Panel title="no branch">
                <Empty>select a branch</Empty>
              </Panel>
            ) : view === "timeline" ? (
              timeline.error ? (
                <Panel title="timeline">
                  <ErrorNote>{timeline.error}</ErrorNote>
                </Panel>
              ) : timeline.data ? (
                <TimelineScrubber
                  branch={current.id}
                  branchName={current.name}
                  timeline={timeline.data}
                  refreshKey={refreshKey}
                />
              ) : (
                <Panel title="timeline">
                  <Empty>reading branch history…</Empty>
                </Panel>
              )
            ) : (
              <DecisionInspector
                branch={current.id}
                branchName={current.name}
                refreshKey={refreshKey}
              />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
