"""HTTP layer for the Recall web UI.

A thin FastAPI app over :class:`kernel.memory.MemoryKernel` plus a Server-Sent
Events feed backed by a CockroachDB changefeed. Contains no SQL and no memory
semantics of its own — see :mod:`api.main`.
"""
