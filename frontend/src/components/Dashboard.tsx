import { useEffect, useState } from "react";
import { api, money, pct, type Summary } from "@/lib/api";
import { BarRow, Kpi, Loading, LoadingText, Panel } from "@/components/ui/shell";

export default function Dashboard() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .summary()
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <Loading error={error}>{null}</Loading>;
  if (!data) return <LoadingText />;

  const maxMargin = Math.max(...data.category_margins.map((c) => Math.abs(c.margin)), 0.0001);
  const maxRegion = Math.max(...data.region_profit_per_sale.map((r) => r.profit_per_sale), 0.0001);
  const p = data.kruskal_p_overall;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        <Kpi label="Revenue" value={money(data.total_revenue)} />
        <Kpi
          label="Lost to cancels"
          value={pct(data.lost_revenue_share)}
          note={`${money(data.lost_revenue_value)} of revenue`}
        />
        <Kpi
          label="Loss-making sales"
          value={pct(data.loss_sale_share)}
          note={`${money(data.loss_sale_count)} sales`}
        />
        <Kpi label="Below-cost products" value={money(data.negative_margin_products)} note="cost >= list price" />
        <Kpi
          label="Stale pending"
          value={money(data.stale_pending_count)}
          note={`${money(data.stale_pending_value)} at risk`}
        />
        <Kpi label="Stock at risk" value={money(data.at_risk)} note={`${data.true_stockout} true stockouts`} />
      </div>

      <Panel title="Worst 10 products by completed profit" hint="loss makers to fix first">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 font-medium">Product</th>
                <th className="py-2 pr-3 font-medium">Category</th>
                <th className="py-2 pr-3 font-medium">Completed profit</th>
                <th className="py-2 font-medium">Sales</th>
              </tr>
            </thead>
            <tbody>
              {data.worst_products.map((r) => (
                <tr key={r.product_id} className="border-t border-border">
                  <td className="py-2 pr-3 font-medium text-foreground">{r.product_id}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{r.category}</td>
                  <td className={`py-2 pr-3 tabular-nums ${r.total_completed_profit < 0 ? "text-destructive" : "text-foreground"}`}>
                    {money(r.total_completed_profit)}
                  </td>
                  <td className="py-2 tabular-nums text-muted-foreground">{money(r.sales)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Category margin" hint="completed profit / completed revenue">
          {data.category_margins.map((c) => (
            <BarRow
              key={c.category}
              label={c.category}
              value={c.margin}
              max={maxMargin}
              display={pct(c.margin)}
              tone={c.margin < 0 ? "up" : "primary"}
            />
          ))}
        </Panel>

        <Panel title="Region profit per sale">
          {data.region_profit_per_sale.map((r) => (
            <BarRow
              key={r.region}
              label={r.region}
              value={r.profit_per_sale}
              max={maxRegion}
              display={money(r.profit_per_sale)}
            />
          ))}
          <p className="mt-2 text-[11px] text-muted-foreground">
            differences not significant (p = {p === null ? "n/a" : p.toFixed(2)})
          </p>
        </Panel>
      </div>
    </div>
  );
}
