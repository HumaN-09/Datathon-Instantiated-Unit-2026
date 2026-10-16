import { useEffect, useState } from "react";
import { CheckCircle2, Maximize2, XCircle } from "lucide-react";
import { api, CHART_GROUPS, CHART_NOTES, money, num, type Metrics } from "@/lib/api";
import { Loading, LoadingText, Panel } from "@/components/ui/shell";
import { LayoutGrid, type GridCard } from "@/components/ui/layout-grid";

/**
 * Stable, page-wide unique card ids: the LayoutGrid uses them as framer-motion
 * layoutIds, so two panels must never share one or the open animation would
 * jump between charts.
 */
const CHART_ORDER = CHART_GROUPS.flatMap((g) => g.names.map((name) => ({ name, group: g.title })));
const CHART_ID: Record<string, number> = Object.fromEntries(
  CHART_ORDER.map((c, i) => [c.name, i + 1]),
);
/** Mosaic rhythm from the component demo: one wide tile, then two square ones. */
const SPANS = ["md:col-span-2", "col-span-1", "col-span-1"];

function MetricTable({
  title,
  data,
  keys,
  hint,
}: {
  title: string;
  data: Record<string, Record<string, number>>;
  keys: string[];
  hint?: string;
}) {
  return (
    <Panel title={title} hint={hint}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-muted-foreground">
            <tr>
              <th className="py-2 pr-3 font-medium">Model</th>
              {keys.map((k) => (
                <th key={k} className="py-2 pr-3 font-medium">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(data).map(([model, vals]) => (
              <tr key={model} className="border-t border-border">
                <td className="py-2 pr-3 font-medium text-foreground">{model}</td>
                {keys.map((k) => (
                  <td key={k} className="py-2 pr-3 tabular-nums text-muted-foreground">
                    {num(vals[k]) === null ? "-" : Number(vals[k]).toFixed(4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

export default function Evidence() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .metrics()
      .then(setM)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <Loading error={error}>{null}</Loading>;
  if (!m) return <LoadingText />;

  const allPass = m.self_checks_all_pass === true;

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="Model registry"
        hint={m.not_trained ? `not trained: ${m.not_trained}` : undefined}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                {["model", "task", "dataset", "features", "target", "metric", "role"].map((h) => (
                  <th key={h} className="py-2 pr-3 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(m.model_registry ?? []).map((r) => (
                <tr key={r.model} className="border-t border-border">
                  <td className="py-2 pr-3 font-medium text-foreground">{r.model}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{r.task}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{r.dataset}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{r.features}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{r.target}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{r.metric}</td>
                  <td className="py-2 pr-3 text-foreground">{r.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Self-check" hint={allPass ? "all checks pass" : "a check failed"}>
        <ul className="flex flex-col gap-2">
          {(m.self_checks ?? []).map((c) => (
            <li key={c.check} className="flex items-start gap-2 text-xs">
              {c.result === "PASS" ? (
                <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
              ) : (
                <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />
              )}
              <span className={c.result === "PASS" ? "text-foreground" : "text-destructive"}>
                {c.result} - {c.check}
              </span>
            </li>
          ))}
        </ul>
      </Panel>

      {m.profit_regression ? (
        <MetricTable
          title="A) Profit regression"
          data={m.profit_regression}
          keys={["r2_train", "r2_test", "mae_test", "rmse_test", "baseline_mae"]}
          hint={m.dataset ? `train ${m.dataset.train_rows} rows, test ${m.dataset.test_rows} rows` : undefined}
        />
      ) : null}

      {m.winner_region ? (
        <MetricTable
          title="B) Winning region"
          data={m.winner_region.models}
          keys={["accuracy", "macro_f1", "log_loss", "uniform_log_loss", "majority_baseline"]}
          hint={`signal: ${m.winner_region.signal} | chance ${m.winner_region.chance.toFixed(2)} | best ${m.winner_region.best_test_accuracy.toFixed(3)} | majority ${m.winner_region.majority_baseline.toFixed(3)}`}
        />
      ) : null}

      {m.region_tests ? (
        <Panel title="C) Kruskal-Wallis on profit across regions" hint="p >= 0.05 means tied">
          <p className="text-xs text-muted-foreground">
            overall p = {m.region_tests.overall.toFixed(4)}
          </p>
          <div className="mt-2 grid grid-cols-2 gap-x-6 sm:grid-cols-3">
            {Object.entries(m.region_tests.by_category).map(([cat, p]) => (
              <div key={cat} className="flex items-baseline justify-between border-b border-border py-1.5">
                <span className="text-xs text-muted-foreground">{cat}</span>
                <span className="text-xs tabular-nums text-foreground">{p.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      {m.shap ? (
        <Panel title="SHAP identity check">
          <p className="text-xs text-muted-foreground">
            base value {money(m.shap.base_value)} + sum of SHAP reconciles to the prediction within{" "}
            <span className="text-foreground">{m.shap.max_base_plus_shap_error.toFixed(6)}</span> across{" "}
            {money(m.shap.sample_rows)} sampled rows.
          </p>
        </Panel>
      ) : null}

      {CHART_GROUPS.map((group) => {
        const cards: GridCard[] = group.names.map((name, i) => {
          const note = CHART_NOTES[name];
          const title = note?.title ?? name.replace(/_/g, " ");
          return {
            id: CHART_ID[name],
            className: `${SPANS[i % SPANS.length]} min-h-[220px]`,
            thumbnail: `/charts/${name}.png`,
            alt: title,
            fit: "contain",
            label: title,
            badge: <Maximize2 className="h-3.5 w-3.5" />,
            content: (
              <div>
                <p className="text-lg font-bold text-white md:text-2xl">{title}</p>
                <p className="mt-0.5 text-[11px] uppercase tracking-wide text-neutral-300">
                  {group.title} / {name}.png
                </p>
                {note ? (
                  <p className="my-2 max-w-lg text-xs text-neutral-200 md:text-sm">{note.note}</p>
                ) : null}
              </div>
            ),
          };
        });
        return (
          <Panel key={group.title} title={group.title} hint="saved at train time - click a chart">
            <LayoutGrid cards={cards} className="max-w-none gap-3 p-0" />
          </Panel>
        );
      })}
    </div>
  );
}
