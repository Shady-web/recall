# Migrations

SQL migration files for the Recall schema land here in **Phase 1**.

Convention (to be established in Phase 1):

- Numbered, forward-only files: `0001_init.sql`, `0002_....sql`, …
- Each migration is idempotent where practical and applied in order.
- The schema (`branches`, `memories`, `decisions`, `decision_memories`,
  `audit_log`, plus the vector index on `memories.embedding`) is defined here —
  see `CONTEXT.md` §7 for the target shape.

No schema is defined yet: Phase 0 is scaffolding only.
