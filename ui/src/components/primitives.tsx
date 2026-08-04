/**
 * The shared vocabulary of the Cold Forensics system.
 *
 * Everything visible in Recall is assembled from these. They are written
 * against the tokens in `styles/tokens.css` from the start — there is no
 * generic layer underneath that was restyled later.
 */

import type { ReactNode } from "react";

/* -------------------------------------------------------------------------
   Instrument labels
   ------------------------------------------------------------------------- */

export function Legend({
  children,
  dim = false,
  className = "",
}: {
  children: ReactNode;
  dim?: boolean;
  className?: string;
}) {
  return (
    <span className={`legend ${dim ? "legend-dim" : ""} ${className}`}>{children}</span>
  );
}

export function Numeric({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <span className={`numeric ${className}`}>{children}</span>;
}

/* -------------------------------------------------------------------------
   Surfaces
   ------------------------------------------------------------------------- */

export function Panel({
  title,
  meta,
  actions,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: string;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-md border border-seam bg-substrate ${className}`}
    >
      {title && (
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-seam px-4 py-2.5">
          <div className="flex min-w-0 items-baseline gap-3">
            <Legend className="text-quartz">{title}</Legend>
            {meta && <span className="truncate">{meta}</span>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

/* -------------------------------------------------------------------------
   Status
   ------------------------------------------------------------------------- */

export type Tone = "live" | "fork" | "retract" | "learned" | "neutral";

const TONE_BORDER: Record<Tone, string> = {
  live: "border-live/40 bg-live-wash text-live",
  fork: "border-fork/40 bg-fork-wash text-fork",
  retract: "border-retract/50 bg-retract-wash text-retract",
  learned: "border-learned/40 bg-learned-wash text-learned",
  neutral: "border-seam-lit bg-strata text-vapor",
};

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: Tone;
  children: ReactNode;
}) {
  return (
    <span
      className={`legend inline-flex items-center gap-1.5 rounded-xs border px-1.5 py-0.5 ${TONE_BORDER[tone]}`}
    >
      {children}
    </span>
  );
}

export function Dot({ tone = "neutral", pulse = false }: { tone?: Tone; pulse?: boolean }) {
  const fill: Record<Tone, string> = {
    live: "bg-live",
    fork: "bg-fork",
    retract: "bg-retract",
    learned: "bg-learned",
    neutral: "bg-graphite",
  };
  return (
    <span
      className={`inline-block size-1.5 shrink-0 rounded-full ${fill[tone]} ${pulse ? "pulse" : ""}`}
    />
  );
}

/* -------------------------------------------------------------------------
   Controls
   ------------------------------------------------------------------------- */

export function Button({
  children,
  onClick,
  disabled = false,
  tone = "neutral",
  title,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tone?: Tone;
  title?: string;
  className?: string;
}) {
  // Accent colours are data, not chrome: even an emphasised button keeps a
  // neutral fill and borrows the accent only for its border and text.
  const emphasis =
    tone === "neutral"
      ? "border-seam-lit bg-strata text-quartz hover:border-graphite hover:text-signal"
      : `${TONE_BORDER[tone]} hover:brightness-125`;
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`legend cursor-pointer rounded-xs border px-2.5 py-1.5 transition-colors duration-150 disabled:cursor-not-allowed disabled:border-seam disabled:bg-transparent disabled:text-graphite ${emphasis} ${className}`}
    >
      {children}
    </button>
  );
}

/* -------------------------------------------------------------------------
   States
   ------------------------------------------------------------------------- */

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full min-h-24 items-center justify-center px-6 py-10 text-center">
      <Legend dim>{children}</Legend>
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="rail rail-retract m-3 rounded-xs bg-retract-wash px-3 py-2">
      <Legend className="text-retract">error</Legend>
      <p className="mt-1 text-xs text-quartz">{children}</p>
    </div>
  );
}

/* -------------------------------------------------------------------------
   Formatting — timestamps are instrument readouts, so always mono and precise
   ------------------------------------------------------------------------- */

export function clock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toISOString().replace("T", " ").slice(0, 19) + "Z";
}

export function timeOnly(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toISOString().slice(11, 23) + "Z";
}

export function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : "—";
}
