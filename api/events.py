"""Live memory-write feed for the UI.

CONTEXT.md §3 lists changefeeds as one of the CockroachDB capabilities Recall
leans on, and this is where that claim is cashed. A background thread holds a
**core changefeed** open —

    EXPERIMENTAL CHANGEFEED FOR memories, decisions, branches
        WITH updated, resolved='2s', no_initial_scan

— which streams row changes back over the SQL connection, and every row is
forwarded to connected browsers over Server-Sent Events. So a memory written by
the agent (or by anyone else touching the cluster) appears on screen without the
UI asking.

**The fallback is not a silent one.** If the changefeed cannot start — rangefeeds
disabled, an insufficiently privileged role, an older cluster — the broker falls
back to polling ``created_at`` and says so: the ``mode`` field on the ``hello``
event is ``"changefeed"`` or ``"poll"``, and the UI renders that verbatim. A demo
that claims a push feed while quietly polling would be exactly the kind of
dishonesty the rest of this project refuses.

Two details worth knowing:

* ``no_initial_scan`` matters. Without it the changefeed backfills every existing
  row on start, so opening the page would replay the entire seeded corpus as if
  it had just been written.
* The ``embedding`` column is stripped before publishing. It is 1024 floats per
  memory — about 20 KB of JSON that no client needs and that would dominate the
  stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

import psycopg

logger = logging.getLogger("recall.api.events")

#: Tables whose changes are streamed to the UI.
WATCHED_TABLES = ("memories", "decisions", "branches")

#: Wire event name per table.
_EVENT_FOR_TABLE = {
    "memories": "memory.changed",
    "decisions": "decision.changed",
    "branches": "branch.changed",
}

#: Never forwarded to clients (see the module docstring).
_STRIPPED_COLUMNS = ("embedding",)

#: How often the poll fallback looks for new rows.
POLL_INTERVAL_SECONDS = 2.0

#: Idle gap after which the SSE endpoint emits a comment to keep proxies from
#: closing the connection.
HEARTBEAT_SECONDS = 15.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


class EventBroker:
    """Fans cluster changes out to any number of SSE subscribers.

    Producers run on plain threads (psycopg is synchronous); consumers are
    asyncio tasks. The two are bridged with ``loop.call_soon_threadsafe``, which
    is why the broker needs the running loop handed to it at startup.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.mode: str = "starting"
        self.detail: str = ""
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._published = 0

    # -- lifecycle --------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run, name="recall-changefeed", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "detail": self.detail,
            "subscribers": len(self._subscribers),
            "published": self._published,
            "watched_tables": list(WATCHED_TABLES),
        }

    # -- publish / subscribe ---------------------------------------------

    def publish(self, event: dict[str, Any]) -> None:
        """Forward one event to every subscriber. Safe to call from any thread."""
        self._published += 1
        loop = self._loop
        if loop is None:
            return
        for queue in list(self._subscribers):
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._offer, queue, event)

    @staticmethod
    def _offer(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        # A subscriber that cannot keep up loses the oldest event rather than
        # stalling the producer — the UI refetches on any event, so a dropped
        # one costs nothing.
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(event)

    @contextlib.contextmanager
    def subscribe(self):
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    # -- producers --------------------------------------------------------

    def _run(self) -> None:
        """Try the changefeed; on failure, fall back to polling and say so."""
        try:
            self._run_changefeed()
        except Exception as exc:
            self.mode = "poll"
            self.detail = f"changefeed unavailable ({type(exc).__name__}: {exc})"
            logger.warning(
                "changefeed could not run (%s: %s); falling back to polling every "
                "%.1fs. The UI will report mode=poll.",
                type(exc).__name__,
                exc,
                POLL_INTERVAL_SECONDS,
            )
            self.publish(
                {
                    "type": "feed.degraded",
                    "at": _now(),
                    "data": {"mode": "poll", "reason": self.detail},
                }
            )
            with contextlib.suppress(Exception):
                self._run_poller()

    def _run_changefeed(self) -> None:
        tables = ", ".join(WATCHED_TABLES)
        # A short `resolved` interval is not about latency for row changes —
        # those arrive immediately. It is how fast the feed can *prove* it is
        # alive on a quiet cluster, which is what /api/health reports and what
        # run_demo.sh waits for before printing the mode.
        sql = (
            f"EXPERIMENTAL CHANGEFEED FOR {tables} "
            f"WITH updated, resolved='2s', no_initial_scan"
        )
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                stream = cur.stream(sql)

                # psycopg sends the statement on the first pull, so a rejected
                # changefeed raises here and the caller demotes us to polling.
                # Getting past the first item means the feed is genuinely open,
                # and that is the moment to say so — waiting for a row would
                # leave /api/health reporting "starting" on a quiet cluster.
                for table, _key, value in stream:
                    if self.mode != "changefeed":
                        self.mode = "changefeed"
                        self.detail = f"core changefeed on {tables}"
                        logger.info("changefeed live on %s", tables)
                    if self._stop.is_set():
                        return
                    # `table is None` is a `resolved` checkpoint: proof of life
                    # with nothing to report.
                    if table is None:
                        continue
                    event = self._event_from_row(table, value)
                    if event is not None:
                        self.publish(event)

    def _event_from_row(self, table: Any, value: Any) -> dict[str, Any] | None:
        name = table.decode() if isinstance(table, bytes) else str(table)
        name = name.split(".")[-1].strip('"')
        event_type = _EVENT_FOR_TABLE.get(name)
        if event_type is None:
            return None
        try:
            payload = json.loads(value.decode() if isinstance(value, bytes) else value)
        except (json.JSONDecodeError, AttributeError):
            return None
        after = payload.get("after")
        if after is None:
            return None
        for column in _STRIPPED_COLUMNS:
            after.pop(column, None)
        return {
            "type": event_type,
            "source": "changefeed",
            "at": _now(),
            "data": after,
        }

    def _run_poller(self) -> None:
        """Fallback: emit rows whose ``created_at`` is newer than last seen.

        TODO(phase-7): this only sees inserts. An in-place status change
        (supersede/retract on the row's own branch) does not move ``created_at``
        and so is missed here — the changefeed path catches it correctly. The UI
        refetches the visible branch on any event, so a retraction is still
        picked up as soon as *anything* else is written; it is only a lone
        retraction with no accompanying insert that goes unnoticed until the
        next user interaction.
        """
        watermarks: dict[str, datetime | None] = dict.fromkeys(WATCHED_TABLES)
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            for table in WATCHED_TABLES:
                row = conn.execute(f"SELECT max(created_at) FROM {table}").fetchone()
                watermarks[table] = row[0] if row else None

            while not self._stop.is_set():
                time.sleep(POLL_INTERVAL_SECONDS)
                for table in WATCHED_TABLES:
                    # One query shape for both cases: an empty table starts at
                    # the epoch rather than taking a separate branch, so the
                    # column metadata always comes from a cursor that ran.
                    mark = watermarks[table]
                    cur = conn.execute(
                        f"SELECT * FROM {table} "
                        f"WHERE created_at > COALESCE(%s, '-infinity'::TIMESTAMPTZ) "
                        f"ORDER BY created_at",
                        (mark,),
                    )
                    columns = [d.name for d in cur.description or []]
                    rows = cur.fetchall()
                    for row in rows:
                        record = dict(zip(columns, row, strict=False))
                        for column in _STRIPPED_COLUMNS:
                            record.pop(column, None)
                        created = record.get("created_at")
                        if created is not None:
                            watermarks[table] = created
                        self.publish(
                            {
                                "type": _EVENT_FOR_TABLE[table],
                                "source": "poll",
                                "at": _now(),
                                "data": json.loads(
                                    json.dumps(record, default=str)
                                ),
                            }
                        )


def sse(event: dict[str, Any]) -> str:
    """Format one event as an SSE frame."""
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
