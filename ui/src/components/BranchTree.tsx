/**
 * Branch tree — main plus every fork, drawn as lanes.
 *
 * Not an org chart and not rounded pills: a git-graph-style lane structure with
 * square nodes and 1px connectors, because a branch is a lineage and the shape
 * should say so. `main` is the root lane and is visually distinct from forks —
 * it carries the live accent, forks carry the fork accent, and a committed or
 * discarded branch drops to a neutral rail because it is no longer live memory.
 *
 * Every count shown is `memory_count` / `decision_count` straight off the API,
 * which aggregates them in SQL. Nothing here is computed client-side.
 */

import type { Branch } from "../api";
import { Badge, Dot, Empty, Legend, Numeric, clock, shortId } from "./primitives";

interface Node {
  branch: Branch;
  depth: number;
  isLast: boolean;
}

/** Flatten the parent/child relation into ordered lanes, roots first. */
function toLanes(branches: Branch[]): Node[] {
  const byParent = new Map<string | null, Branch[]>();
  for (const branch of branches) {
    const key = branch.parent_branch_id;
    const bucket = byParent.get(key) ?? [];
    bucket.push(branch);
    byParent.set(key, bucket);
  }

  const known = new Set(branches.map((b) => b.id));
  const nodes: Node[] = [];

  const walk = (parent: string | null, depth: number) => {
    const children = byParent.get(parent) ?? [];
    children.forEach((branch, index) => {
      nodes.push({ branch, depth, isLast: index === children.length - 1 });
      walk(branch.id, depth + 1);
    });
  };

  walk(null, 0);
  // A branch whose parent is missing (an orphan) would otherwise vanish
  // silently. Surface it at the root rather than dropping it.
  for (const branch of branches) {
    if (branch.parent_branch_id && !known.has(branch.parent_branch_id)) {
      nodes.push({ branch, depth: 0, isLast: true });
    }
  }
  return nodes;
}

function railFor(branch: Branch): string {
  if (branch.status !== "open") return "rail-absent";
  return branch.parent_branch_id === null ? "rail-active" : "rail-fork";
}

export function BranchTree({
  branches,
  selected,
  onSelect,
}: {
  branches: Branch[];
  selected: string | null;
  onSelect: (branch: Branch) => void;
}) {
  if (branches.length === 0) {
    return <Empty>no branches — run the seed script</Empty>;
  }

  const lanes = toLanes(branches);

  return (
    <ul className="flex flex-col gap-px overflow-y-auto p-2">
      {lanes.map(({ branch, depth }) => {
        const isSelected = branch.id === selected;
        const isRoot = branch.parent_branch_id === null;
        return (
          <li key={branch.id} style={{ paddingLeft: depth * 14 }}>
            <button
              type="button"
              onClick={() => onSelect(branch)}
              aria-current={isSelected}
              className={`rail ${railFor(branch)} group w-full cursor-pointer rounded-xs px-2.5 py-2 text-left transition-colors duration-150 ${
                isSelected ? "bg-strata" : "hover:bg-strata/60"
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span
                  className={`truncate font-mono text-xs ${
                    isSelected ? "text-signal" : "text-quartz group-hover:text-signal"
                  }`}
                  title={branch.name}
                >
                  {isRoot ? branch.name : branch.name.replace(/^incident\//, "")}
                </span>
                {branch.status !== "open" && (
                  <Badge tone="neutral">{branch.status}</Badge>
                )}
              </div>

              <div className="mt-1 flex items-center gap-2.5">
                {isRoot ? (
                  <span className="flex items-center gap-1.5">
                    <Dot tone="live" />
                    <Legend dim>root</Legend>
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5">
                    <Dot tone={branch.status === "open" ? "fork" : "neutral"} />
                    <Legend dim>fork</Legend>
                  </span>
                )}
                <Numeric className="text-vapor">{branch.memory_count} mem</Numeric>
                {branch.decision_count > 0 && (
                  <Numeric className="text-fork">{branch.decision_count} dec</Numeric>
                )}
              </div>

              <div className="mt-1">
                <Numeric className="text-graphite">
                  {shortId(branch.id)} · {clock(branch.fork_point_ts ?? branch.created_at)}
                </Numeric>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
