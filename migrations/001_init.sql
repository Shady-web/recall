-- 001_init.sql — Recall schema, Phase 1 (schema + non-semantic kernel core).
--
-- Implements the tables in CONTEXT.md §7. Explicit design decisions (per the
-- Phase 1 brief) are documented inline:
--
--  * PRIMARY KEYs are UUIDs generated with gen_random_uuid(). UUIDs let
--    concurrent agents — and, later, forked branches — mint ids without
--    coordination, which matters once branching and replay land.
--
--  * Every table has created_at TIMESTAMPTZ NOT NULL DEFAULT now(). CONTEXT §7
--    named the audit timestamp `ts`; we unify on created_at across ALL tables
--    per this phase's explicit decision. For audit_log, created_at IS the event
--    timestamp.
--
--  * memories.embedding is VECTOR(1024). 1024 matches Amazon Titan Text
--    Embeddings V2's native output dimension, and was verified to be both
--    storable AND indexable on CockroachDB v25.2 (the vector index is gated
--    behind the feature.vector_index.enabled cluster setting, but a
--    1024-dimension CREATE VECTOR INDEX succeeds). The column is left NULLABLE
--    and UNINDEXED here — embeddings and the vector index are populated in
--    Phase 2, so building the index now would be speculative.
--      refs: https://www.cockroachlabs.com/docs/v25.2/vector
--            https://www.cockroachlabs.com/docs/v25.2/vector-indexes
--
--  * Foreign keys use ON DELETE RESTRICT wherever a delete would destroy the
--    system-of-record or its provenance. The single CASCADE is on the pure
--    child rows of a decision. Each choice is justified at the column.
--
--  * Indexes cover only the access paths the kernel exercises in this phase.
--
--  * This file is idempotent (IF NOT EXISTS / ON CONFLICT) so re-running it — or
--    the migration runner — is safe.

-- branches --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS branches (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             STRING NOT NULL,
    -- Self-reference up the lineage. NULL only for the root branch.
    -- RESTRICT: a branch with descendants (or memories) must not be hard-deleted
    -- — lineage is part of the audit story. "Discarding" a branch is a status
    -- change (Phase 3), never a row delete.
    parent_branch_id UUID REFERENCES branches (id) ON DELETE RESTRICT,
    -- Point in the parent's history this branch forked from. NULL for the root;
    -- populated by the branching engine in Phase 3.
    fork_point_ts    TIMESTAMPTZ,
    status           STRING NOT NULL DEFAULT 'open'
                     CHECK (status IN ('open', 'committed', 'discarded')),
    created_by       STRING NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Branches are resolved by name (e.g. 'main'). One unique index both enforces
-- name uniqueness and serves the name lookup.
CREATE UNIQUE INDEX IF NOT EXISTS idx_branches_name ON branches (name);

-- memories --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- RESTRICT: memories are the system of record. Deleting a branch must never
    -- silently delete the facts recorded against it.
    branch_id     UUID NOT NULL REFERENCES branches (id) ON DELETE RESTRICT,
    kind          STRING NOT NULL,
    content       STRING NOT NULL,
    -- Populated in Phase 2. See dimension rationale in the header.
    embedding     VECTOR(1024),
    source        STRING,
    confidence    FLOAT8 NOT NULL DEFAULT 1.0
                  CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status        STRING NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'superseded', 'retracted')),
    -- When a memory is superseded this points at its replacement. We never
    -- delete the old row; supersede() links old -> new. RESTRICT so the
    -- replacement cannot be deleted out from under the history chain.
    superseded_by UUID REFERENCES memories (id) ON DELETE RESTRICT,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- list_memories() always filters by branch_id and orders by created_at DESC
-- (newest first) with limit/offset. This index is that exact access path.
CREATE INDEX IF NOT EXISTS idx_memories_branch_created
    ON memories (branch_id, created_at DESC);

-- decisions -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- RESTRICT: decisions are audit records; a branch delete must not erase them.
    branch_id  UUID NOT NULL REFERENCES branches (id) ON DELETE RESTRICT,
    agent_id   STRING NOT NULL,
    -- input_hash and outcome are filled in by later phases; nullable for now.
    input_hash STRING,
    action     STRING NOT NULL,
    rationale  STRING,
    outcome    STRING,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Decisions are listed/inspected per branch, newest first.
CREATE INDEX IF NOT EXISTS idx_decisions_branch_created
    ON decisions (branch_id, created_at DESC);

-- decision_memories -----------------------------------------------------------
-- Provenance: which memories were recalled to produce a decision.
CREATE TABLE IF NOT EXISTS decision_memories (
    -- CASCADE: these rows have no meaning without their decision — they are pure
    -- children of exactly one decision. If a decision were ever removed, its
    -- provenance rows must go with it (no orphans).
    decision_id UUID NOT NULL REFERENCES decisions (id) ON DELETE CASCADE,
    -- RESTRICT: a memory cited as provenance must not be deletable — that would
    -- silently break the "which memory drove this decision" guarantee.
    memory_id   UUID NOT NULL REFERENCES memories (id) ON DELETE RESTRICT,
    -- similarity is populated once recall is vector-based (Phase 2); nullable now.
    similarity  FLOAT8,
    rank        INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (decision_id, memory_id)
);

-- audit_log -------------------------------------------------------------------
-- Append-only log of every kernel operation (read, write, fork, commit). Every
-- kernel op writes exactly one row here INSIDE the same transaction as the op.
CREATE TABLE IF NOT EXISTS audit_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor       STRING NOT NULL,
    op          STRING NOT NULL,
    target_type STRING NOT NULL,
    target_id   UUID,
    payload     JSONB NOT NULL DEFAULT '{}',
    -- CONTEXT §7 called this `ts`; unified to created_at (see header).
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Audit trails are read as "everything that happened to object X", so index by
-- the target, newest first.
CREATE INDEX IF NOT EXISTS idx_audit_target
    ON audit_log (target_type, target_id, created_at DESC);

-- Seed the root branch --------------------------------------------------------
-- 'main' is the root: no parent, no fork point. Idempotent via ON CONFLICT on
-- the unique name, so re-running the migration is a no-op.
INSERT INTO branches (name, parent_branch_id, fork_point_ts, status, created_by)
VALUES ('main', NULL, NULL, 'open', 'system')
ON CONFLICT (name) DO NOTHING;
