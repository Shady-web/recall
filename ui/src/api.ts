/**
 * The only place the UI reaches the network.
 *
 * Every field rendered anywhere in this app comes through one of these calls,
 * which means it came from the kernel, which means it came from CockroachDB.
 * There are no seeded constants, no computed-looking placeholders, and no
 * "example" values in the client. If a number is on screen, it was read.
 */

export type BranchStatus = "open" | "committed" | "discarded";

export interface Branch {
  id: string;
  name: string;
  parent_branch_id: string | null;
  fork_point_ts: string | null;
  status: BranchStatus;
  created_by: string;
  created_at: string;
  memory_count: number;
  decision_count: number;
  child_count: number;
}

export interface Memory {
  id: string;
  branch_id: string;
  kind: string;
  content: string;
  source: string | null;
  confidence: number;
  status: "active" | "superseded" | "retracted";
  superseded_by: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface MemoryRef {
  memory_id: string;
  content: string;
  kind: string;
  status: string;
}

export interface Decision {
  id: string;
  branch_id: string;
  agent_id: string;
  action: string;
  rationale: string | null;
  outcome: string | null;
  created_at: string;
}

export interface ContributingMemory {
  memory_id: string;
  content: string;
  kind: string;
  source: string | null;
  confidence: number;
  branch_id: string;
  created_at: string;
  similarity: number | null;
  rank: number;
  status_now: string;
  invalidated: boolean;
  superseded: boolean;
  retracted: boolean;
  superseded_at: string | null;
  retracted_at: string | null;
}

export interface DecisionExplanation {
  decision: Decision;
  branch_id: string;
  branch_name: string;
  memories: ContributingMemory[];
  has_invalidated_memories: boolean;
  invalidated_count: number;
  invalidated_memory_ids: string[];
}

export interface MemoryAvailabilityDiff {
  only_then: MemoryRef[];
  only_now: MemoryRef[];
  common_count: number;
  then_count: number;
  now_count: number;
}

export interface RerunResult {
  decision_id: string;
  branch_name: string;
  decision_at: string;
  evaluated_at: string;
  old_action: string;
  old_rationale: string | null;
  new_action: string;
  new_rationale: string | null;
  action_changed: boolean;
  memory_diff: MemoryAvailabilityDiff;
  contributing_memory_ids: string[];
}

export interface RerunPair {
  decision_id: string;
  reasoner: string;
  faithful: RerunResult;
  today: RerunResult;
  verdict_changed: boolean;
}

export interface ReplayWindow {
  gc_ttl_seconds: number;
  earliest: string;
  latest: string;
}

export interface Timeline {
  branch: Branch | null;
  earliest: string;
  latest: string;
  fork_point_ts: string | null;
  memory_count: number;
  replay_window: ReplayWindow;
}

export interface Health {
  cluster_reachable: boolean;
  cluster_version: string | null;
  actor: string;
  read_only: boolean;
  offline_mode: boolean;
  embedding_provider: string;
  reasoning_provider: string;
  live_feed: {
    mode: "starting" | "changefeed" | "poll";
    detail: string;
    subscribers: number;
    published: number;
    watched_tables: string[];
  };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly kind: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let kind = "HTTPError";
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      const detail = body?.detail;
      if (detail && typeof detail === "object") {
        kind = detail.type ?? kind;
        message = detail.message ?? message;
      } else if (typeof detail === "string") {
        message = detail;
      }
    } catch {
      /* a non-JSON error body is still an error; keep the status line */
    }
    throw new ApiError(response.status, kind, message);
  }
  return response.json() as Promise<T>;
}

const qs = (params: Record<string, string | number | undefined | null>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) search.set(key, String(value));
  }
  return search.toString();
};

export const api = {
  health: () => request<Health>("/health"),

  replayWindow: () => request<ReplayWindow>("/replay-window"),

  branches: () => request<{ count: number; branches: Branch[] }>("/branches"),

  memories: (branch: string, limit = 200) =>
    request<{ count: number; memories: Memory[] }>(`/memories?${qs({ branch, limit })}`),

  timeline: (branch: string) => request<Timeline>(`/timeline?${qs({ branch })}`),

  /** What `branch` knew at instant `t`. `physical` uses AS OF SYSTEM TIME. */
  replay: (branch: string, t: string, mode: "logical" | "physical" = "logical") =>
    request<{ count: number; mode: string; memories: Memory[] }>(
      `/replay?${qs({ branch, t, mode })}`,
    ),

  decisions: (branch?: string) =>
    request<{ count: number; decisions: Decision[] }>(`/decisions?${qs({ branch })}`),

  explain: (id: string) => request<DecisionExplanation>(`/decisions/${id}/explain`),

  /** Runs the agent twice. Only call this on explicit operator action. */
  rerun: (id: string) =>
    request<RerunPair>(`/decisions/${id}/rerun`, { method: "POST" }),

  fireIncident: (incident: string, commit = false) =>
    request<Record<string, unknown>>("/incidents", {
      method: "POST",
      body: JSON.stringify({ incident, commit }),
    }),

  retract: (memoryId: string, reason: string, branch?: string) =>
    request<Memory>(`/memories/${memoryId}/retract`, {
      method: "POST",
      body: JSON.stringify({ reason, branch: branch ?? null }),
    }),

  commit: (branch: string) =>
    request<{ committed: boolean; conflict_count: number }>("/commit", {
      method: "POST",
      body: JSON.stringify({ branch }),
    }),
};
