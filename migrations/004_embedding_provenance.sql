-- 004_embedding_provenance.sql — record WHICH provider produced each embedding.
--
-- WHY THIS MIGRATION EXISTS
--
-- A vector is only comparable to vectors from the same embedding space. Recall
-- ships two providers on purpose — Amazon Titan for real use, and a
-- deterministic fake one so the suite and the demo can run at zero cost — and
-- both emit 1024-dimension unit vectors. That makes them structurally
-- indistinguishable and silently incomparable.
--
-- The failure this prevents was observed, not imagined. A database seeded with
-- the fake provider was then queried with Titan. Nothing errored. Recall
-- returned six confident-looking hits with similarity scores of 0.040, 0.022,
-- 0.008, 0.001, -0.009, -0.011 — which is precisely what cosine similarity
-- looks like between two unrelated vector spaces: orthogonal noise. The ranking
-- was meaningless, and the only clue was that the numbers looked "a bit low".
--
-- That is the worst class of bug in a memory system: it does not fail, it just
-- quietly stops being about meaning. And the project's own demo workflow walks
-- straight into it — seed cheaply offline, then record against real Bedrock.
--
-- The existing guard (kernel.db.verify_embedding_dimension) cannot catch this:
-- both providers agree on width. Width was never the invariant that mattered;
-- the SPACE is. This column records the space, so a mismatch can be detected
-- and refused up front instead of being served as plausible nonsense.
--
-- Nullable, because rows written before this migration have an unknown
-- provenance and we will not guess one for them. The verifier treats NULL as
-- "unverifiable" and warns rather than failing, so an existing database keeps
-- working while new writes become self-describing.

ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_model STRING;

-- The verifier reads the distinct set of providers present on the rows a branch
-- can see, so this is a low-cardinality grouping scan over a column that is
-- almost always one value. Indexed to keep that check cheap as the corpus grows.
CREATE INDEX IF NOT EXISTS idx_memories_embedding_model
    ON memories (embedding_model);
