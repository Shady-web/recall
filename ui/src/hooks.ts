/**
 * Data hooks: fetching, the live feed, and the scrubber's motion smoothing.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

/** A single async read with explicit loading/error state and manual refetch. */
export function useAsync<T>(
  load: () => Promise<T>,
  deps: unknown[],
): {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loadRef
      .current()
      .then((value) => {
        if (cancelled) return;
        setData(value);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, error, loading, reload: useCallback(() => setNonce((n) => n + 1), []) };
}

export interface LiveEvent {
  type: string;
  source?: string;
  at?: string | null;
  data?: Record<string, unknown>;
}

/**
 * Subscribe to the backend's Server-Sent Events feed.
 *
 * `mode` is whatever the server reported in its `hello` frame — `changefeed`
 * when a CockroachDB changefeed is genuinely streaming, `poll` when it fell
 * back. The status bar shows that value directly rather than assuming push
 * worked; a demo that claimed a live feed while polling would be a lie told in
 * the one place this project cares most about telling the truth.
 */
export function useLiveFeed(): {
  connected: boolean;
  mode: string;
  lastEvent: LiveEvent | null;
  eventCount: number;
} {
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState("connecting");
  const [lastEvent, setLastEvent] = useState<LiveEvent | null>(null);
  const [eventCount, setEventCount] = useState(0);

  useEffect(() => {
    const source = new EventSource("/api/events");

    source.addEventListener("open", () => setConnected(true));

    source.addEventListener("hello", (event) => {
      setConnected(true);
      try {
        const payload = JSON.parse((event as MessageEvent).data) as LiveEvent;
        setMode(String(payload.data?.mode ?? "unknown"));
      } catch {
        setMode("unknown");
      }
    });

    const onChange = (event: Event) => {
      try {
        setLastEvent(JSON.parse((event as MessageEvent).data) as LiveEvent);
      } catch {
        /* a malformed frame is not worth tearing the feed down for */
      }
      setEventCount((n) => n + 1);
    };
    for (const name of [
      "memory.changed",
      "decision.changed",
      "branch.changed",
      "feed.degraded",
    ]) {
      source.addEventListener(name, onChange);
    }

    source.addEventListener("error", () => setConnected(false));

    return () => source.close();
  }, []);

  return { connected, mode, lastEvent, eventCount };
}

/**
 * Smoothing for the timeline scrubber.
 *
 * The scrubber must feel continuous while each position implies a database
 * round trip. So the returned `immediate` value tracks the drag with no delay
 * (it drives the playhead and the clock readout), while `settled` lags by
 * `delayMs` and is what triggers the fetch. The result is a control that never
 * stutters and a backend that is not asked a question per pixel.
 */
export function useDebounced<T>(value: T, delayMs = 90): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return settled;
}

/** Keeps the previous non-null value while a new one is in flight. */
export function useLastKnown<T>(value: T | null): T | null {
  const ref = useRef<T | null>(null);
  if (value !== null) ref.current = value;
  return ref.current;
}
