-- 003_branch_ancestry.sql — schema support for branch ancestry resolution (Phase 3).
--
-- WHY THIS MIGRATION EXISTS
--
-- Phase 3 needs "reads on a branch see the parent's state AS OF the fork point".
-- The original design (CONTEXT.md §7) assumed `AS OF SYSTEM TIME` at each fork
-- point inside one recursive CTE. That is NOT implementable on CockroachDB:
--
--   * AOST is a top-level statement (or transaction) modifier. Placing it in a
--     CTE, a sub-select, or on one arm of a UNION fails with
--     "AS OF SYSTEM TIME must be provided on a top-level statement", so a single
--     statement cannot read different ancestry segments at different timestamps.
--   * AOST is bounded by MVCC garbage collection (gc.ttlseconds defaults to
--     14400 = 4h). A branch older than the GC window would become permanently
--     unreadable — fatal for a durable memory system.
--
-- So ancestry uses LOGICAL time-travel: every ancestry segment is bounded by a
-- `created_at <= fork_point` predicate instead of an MVCC timestamp. That keeps
-- the read in ONE SQL statement, keeps the vector index on the hot path, and is
-- not GC-bounded. (AOST remains load-bearing for Phase 4 replay, where a single
-- whole-statement timestamp is exactly the right tool — verified working, and
-- verified compatible with the vector index.)
--
-- Logical time-travel needs two things the Phase 1/2 schema lacked:
--
--  1. WHEN a status change happened. `supersede`/`retract` mutated `status` in
--     place, so a descendant would see the parent's *current* status rather than
--     its status at the fork point. `superseded_at` / `retracted_at` make
--     "status as of T" computable in pure SQL.
--
--  2. A way to change the status of an INHERITED memory without touching the
--     ancestor's row (writes on a branch must never affect the parent). The
--     `memory_overrides` table is a branch-local status overlay: it shadows an
--     ancestor's memory for one branch only.

-- 1. When did a status change happen? ----------------------------------------
ALTER TABLE memories ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS retracted_at  TIMESTAMPTZ;

-- Provenance for rows replayed onto a parent by commit(): points at the branch
-- memory this row was replayed from. RESTRICT because it is audit lineage.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS origin_memory_id UUID
    REFERENCES memories (id) ON DELETE RESTRICT;

-- Backfill for pre-003 rows. Both statements are guarded with `IS NULL`, so
-- re-running this migration is a no-op.
--
-- Superseded rows: the exact supersession time is recoverable — it is when the
-- replacement row was created.
UPDATE memories m
   SET superseded_at = (SELECT n.created_at FROM memories n WHERE n.id = m.superseded_by)
 WHERE m.status = 'superseded'
   AND m.superseded_at IS NULL
   AND m.superseded_by IS NOT NULL;

-- Retracted rows: the retraction time was never recorded before this migration.
-- We bound it at migration time (rather than created_at) so that historical
-- reads before this point still see the memory as it actually was — active.
-- This is an approximation, and it only affects rows written before Phase 3.
UPDATE memories
   SET retracted_at = now()
 WHERE status = 'retracted' AND retracted_at IS NULL;

-- 2. Branch-local status overlay ---------------------------------------------
-- One row per (branch, inherited memory) whose status this branch changed.
-- The ancestor's row is never touched, so the parent is unaffected.
CREATE TABLE IF NOT EXISTS memory_overrides (
    -- RESTRICT throughout: overrides are part of the audit/lineage record.
    branch_id     UUID NOT NULL REFERENCES branches (id) ON DELETE RESTRICT,
    memory_id     UUID NOT NULL REFERENCES memories (id) ON DELETE RESTRICT,
    status        STRING NOT NULL CHECK (status IN ('superseded', 'retracted')),
    -- For a supersede: the branch-local replacement memory.
    superseded_by UUID REFERENCES memories (id) ON DELETE RESTRICT,
    reason        STRING,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- One override per memory per branch. Also the lookup path used by recall:
    -- "does any ancestry branch override this memory?"
    PRIMARY KEY (branch_id, memory_id)
);

-- Recall resolves overrides by memory across the ancestry chain, so index the
-- reverse direction too.
CREATE INDEX IF NOT EXISTS idx_memory_overrides_memory
    ON memory_overrides (memory_id, branch_id);

-- Ancestry walks branches parent-ward; index the FK for the recursive CTE.
CREATE INDEX IF NOT EXISTS idx_branches_parent
    ON branches (parent_branch_id);
