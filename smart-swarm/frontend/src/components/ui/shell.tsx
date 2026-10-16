import { cn } from "@/lib/utils";

/** Panel: the standard semi-transparent surface used inside every tab. */
export function Panel({
  children,
  className,
  title,
  hint,
}: {
  children: React.ReactNode;
  className?: string;
  title?: string;
  hint?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-[var(--radius)] border border-border bg-[rgba(28,28,28,0.66)] p-4 backdrop-blur",
        className,
      )}
    >
      {title ? (
        <div className="mb-3 flex items-baseline justify-between gap-3">
          <h2 className="text-sm font-medium text-foreground">{title}</h2>
          {hint ? <span className="text-[11px] text-muted-foreground">{hint}</span> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

/** Kpi: one number with a label. */
export function Kpi({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-[var(--radius)] border border-border bg-[rgba(28,28,28,0.66)] p-3 backdrop-blur">
      <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular-nums text-foreground">{value}</p>
      {note ? <p className="mt-1 text-[11px] text-muted-foreground">{note}</p> : null}
    </div>
  );
}

/** BarRow: a labelled CSS bar, used for margins, region profit and SHAP. */
export function BarRow({
  label,
  value,
  max,
  display,
  tone = "primary",
}: {
  label: string;
  value: number;
  max: number;
  display: string;
  tone?: "primary" | "up" | "down" | "muted";
}) {
  const width = max > 0 ? Math.min(100, Math.abs(value / max) * 100) : 0;
  const colors: Record<string, string> = {
    primary: "bg-primary",
    up: "bg-[#ef4444]",
    down: "bg-[#4f7cff]",
    muted: "bg-muted",
  };
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 py-1">
      <div className="min-w-0">
        <p className="truncate text-xs text-muted-foreground">{label}</p>
        <div className="mt-1 h-1.5 w-full rounded-full bg-muted">
          <div className={cn("h-1.5 rounded-full", colors[tone])} style={{ width: `${width}%` }} />
        </div>
      </div>
      <span className="text-xs font-medium tabular-nums text-foreground">{display}</span>
    </div>
  );
}

/** Sparkline: smooth line with end dots, drawn as inline SVG. */
export function Sparkline({
  points,
  height = 44,
  className,
}: {
  points: number[];
  height?: number;
  className?: string;
}) {
  const clean = points.map((p) => (Number.isFinite(p) ? p : 0));
  if (clean.length < 2) {
    return <p className="text-xs text-muted-foreground">Not enough history</p>;
  }
  const w = 260;
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const span = max - min || 1;
  const step = w / (clean.length - 1);
  const pts = clean.map((v, i) => [i * step, height - ((v - min) / span) * (height - 8) - 4] as const);
  const d = pts
    .map(([x, y], i) => {
      if (i === 0) return `M ${x.toFixed(1)} ${y.toFixed(1)}`;
      const [px, py] = pts[i - 1];
      const cx = px + step / 2;
      return `C ${cx.toFixed(1)} ${py.toFixed(1)}, ${cx.toFixed(1)} ${y.toFixed(1)}, ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  const last = pts[pts.length - 1];
  const first = pts[0];
  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      className={cn("h-11 w-full", className)}
      preserveAspectRatio="none"
      role="img"
      aria-label="12 month profit trend"
    >
      <path d={d} fill="none" stroke="#d1cfc0" strokeWidth={1.6} />
      <circle cx={first[0]} cy={first[1]} r={2} fill="#8e8a83" />
      <circle cx={last[0]} cy={last[1]} r={2.6} fill="#d1cfc0" />
    </svg>
  );
}

export function Pill({ tone, children }: { tone: "sell" | "hold"; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "rounded-full border px-2 py-0.5 text-[11px] font-medium",
        tone === "sell"
          ? "border-primary/50 text-primary"
          : "border-destructive/50 text-destructive",
      )}
    >
      {children}
    </span>
  );
}

export function Loading({ error, children }: { error: string | null; children: React.ReactNode }) {
  if (error) {
    return (
      <Panel title="Something failed">
        <p className="text-sm text-destructive">{error}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Is the API running? Start it with: uvicorn backend.main:app --port 8000
        </p>
      </Panel>
    );
  }
  return <>{children}</>;
}

export function LoadingText() {
  return <p className="text-sm text-muted-foreground">Loading...</p>;
}
