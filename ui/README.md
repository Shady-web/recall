# Recall Web UI

Placeholder. The demo surface lands in Phase 6.

Planned stack: **React + Vite + Tailwind**.

Planned surfaces:

- **Timeline scrubber** — replay memory state at any past instant.
- **Branch tree** — visualize forks, commits, and discards.
- **Decision inspector** — show which recalled memories drove a given decision
  (provenance), fed live by a CockroachDB changefeed.

Nothing here talks to CockroachDB directly. The UI calls the kernel through the
HTTP layer (API Gateway → FastAPI → kernel).
