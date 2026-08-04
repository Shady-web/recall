# Recall Web UI

React + Vite + Tailwind v4. Three surfaces over the memory kernel, all reading
live.

```bash
# From the repo root — starts the cluster, seeds, backend and UI together:
./scripts/run_demo.sh

# Or just the frontend, against a backend already on :8000:
npm install && npm run dev
```

Nothing here talks to CockroachDB directly. The UI calls the kernel through the
HTTP layer (`api/` → kernel → cluster).

## The honesty rule

Nothing in `src/` contains a seeded value, a placeholder number, or an example
row. Every field on screen arrives through `src/api.ts`, which talks to the
FastAPI bridge, which calls the kernel, which reads CockroachDB. If a count, a
similarity score, a timestamp, or a status is rendered, it was read on that
request.

Two consequences worth keeping:

- The status bar prints the **actual** live-feed mode (`changefeed` or `poll`),
  the **actual** embedding and reasoning providers, and the **actual** cluster
  build string. Running in offline mode is badged loudly, because a fake
  embedder and a rule reasoner must never look like a Bedrock run.
- The forensic-replay control is disabled — visibly, with the reason in its
  tooltip — when the scrubbed instant falls outside the cluster's MVCC garbage
  collection window. The limit is real, so it is shown rather than hidden behind
  a request that would fail.

## Surfaces

| View | What it answers |
|---|---|
| **Branch tree** (always visible) | What lineages exist, which are open/committed/discarded, and what each one contributed |
| **Timeline scrubber** | What did this branch know at instant *T* — and how does that differ from now |
| **Decision inspector** | Which memories drove this decision, are any of them since withdrawn, and would the agent still decide this way |

The rewind contrast in the decision inspector is the point of the whole
project: two agent runs of the same question, one against decision-time memory
and one against today's, rendered side by side. Either panel alone proves
nothing.

## Design system — "Cold Forensics"

The tokens live in [`src/styles/tokens.css`](src/styles/tokens.css), and that
file documents the system in full: what was taken as direction from the Refero
reference (Factory — "terminal war room at midnight"), and the five places this
system deliberately diverges from it.

The short version:

- **Cold, not warm.** A blue-black canvas (`#07090d`) and a slate ramp, against
  the reference's warm near-black and warm greys. This is a tool for reading
  evidence, not a war room.
- **The provenance rail.** Recall's signature element, and its own: a 2px bar
  down the left of every memory and decision row whose colour encodes lifecycle.
  The product's subject is *what a fact's status was then versus now*, so the
  most-repeated visual element answers exactly that without a word being read.
- **Mono labels open up** (`+0.09em`), against the reference's uniformly tight
  tracking — instrument legends stamped on a panel, not compressed display type.
- **Denser and sharper.** A 4px base and 2/6/12px radii, for an application
  showing hundreds of rows.
- **Four accents, one meaning each.** `live` (now / read from cluster), `fork`
  (lineage), `retract` (withdrawn), `learned` (acquired since). More than the
  reference permits, because the four-way distinction *is* the product. The
  discipline is kept: an accent never becomes chrome — even an emphasised button
  keeps a neutral fill and borrows the accent only for border and text.

No drop shadows anywhere. Depth is contrast and spacing.

## Structure

```
src/
  api.ts                     the only place the UI touches the network
  hooks.ts                   async reads, the SSE feed, scrubber smoothing
  App.tsx                    layout, branch selection, live-refresh wiring
  styles/tokens.css          the design system (read this first)
  components/
    primitives.tsx           Panel, Badge, Button, Legend, rails, formatting
    StatusBar.tsx            live cluster facts
    BranchTree.tsx           lineage lanes
    TimelineScrubber.tsx     the then/now scrubber
    DecisionInspector.tsx    provenance + the rewind contrast
```

## Live updates

`hooks.ts:useLiveFeed` subscribes to `/api/events` (Server-Sent Events, fed by a
CockroachDB changefeed). Any change event bumps a `refreshKey` that every query
depends on, so the screen refetches rather than patching local state — the
refetch is what guarantees the view still matches the cluster instead of
matching our idea of it.

## Scrubber smoothness

Each scrubber position implies a replay query, so the control is split: the
playhead and clock follow the drag with zero delay, while the fetch runs off a
90 ms debounce. During a fetch the previous result stays on screen at reduced
opacity rather than blanking, so dragging never flickers or empties.
