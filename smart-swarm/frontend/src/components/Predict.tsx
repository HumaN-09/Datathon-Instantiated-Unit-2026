import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Sparkles } from "lucide-react";
import {
  api,
  money,
  pct,
  text,
  type CampaignCard,
  type CategoryCard,
  type Options,
  type PlanRequest,
  type PredictResult,
  type RegionPlan,
} from "@/lib/api";
import { CardDeckCarousel, type DeckSlide } from "@/components/ui/card-deck-carousel";
import { GlowCard } from "@/components/ui/spotlight-card";
import { BarRow, Loading, LoadingText, Panel, Pill, Sparkline } from "@/components/ui/shell";

const BEST = "Best option (let the app choose)";
const SALES_CHANNELS = ["Web", "App", "Store"];

export default function Predict() {
  const [options, setOptions] = useState<Options | null>(null);
  const [catCards, setCatCards] = useState<CategoryCard[]>([]);
  const [campCards, setCampCards] = useState<CampaignCard[]>([]);
  const [campaignNote, setCampaignNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [category, setCategory] = useState<string | null>(null);
  const [campaignChannel, setCampaignChannel] = useState<string | null>(null);
  const [productId, setProductId] = useState("");
  const [discount, setDiscount] = useState(0.2);
  const [quantity, setQuantity] = useState(2);
  const [segment, setSegment] = useState("Loyal");
  const [month, setMonth] = useState(11);
  const [region, setRegion] = useState<string>(BEST);

  const [result, setResult] = useState<PredictResult | null>(null);
  const [plan, setPlan] = useState<RegionPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.options(), api.categoryCards(), api.campaignCards()])
      .then(([o, c, g]) => {
        setOptions(o);
        setCatCards(c.cards);
        setCampCards(g.cards);
        setCampaignNote(g.note);
        const first = c.cards[0]?.category ?? null;
        setCategory(first);
        setProductId(o.products[0]?.product_id ?? "");
        setSegment(o.customer_segments[0] ?? "Loyal");
        setCampaignChannel(g.cards[0]?.channel ?? null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const products = useMemo(() => {
    if (!options) return [];
    return category ? options.products.filter((p) => p.category === category) : options.products;
  }, [options, category]);

  useEffect(() => {
    if (products.length === 0) return;
    if (!products.some((p) => p.product_id === productId)) {
      setProductId(products[0].product_id);
    }
  }, [products, productId]);

  const ventureSlides: DeckSlide[] = catCards.map((c) => ({
    title: c.title,
    value: c.margin_pct === null ? "n/a" : `${c.margin_pct.toFixed(1)}% margin`,
    caption: `${c.share_products_below_cost === null ? "n/a" : pct(c.share_products_below_cost)} of products priced below cost`,
    meta: `campaign CPA ${money(c.campaign_median_cpa)} | CVR ${c.campaign_median_cvr === null ? "n/a" : pct(c.campaign_median_cvr)}`,
  }));

  const campaignSlides: DeckSlide[] = campCards.map((c) => ({
    title: c.title,
    value: money(c.median_cpa),
    caption: `median CPA ${money(c.median_cpa)} | median CVR ${c.median_cvr === null ? "n/a" : pct(c.median_cvr)}`,
    meta: `${money(c.valid_rows)} valid rows | spend ${money(c.spend)}`,
  }));

  const run = useCallback(
    async (body: PlanRequest) => {
      setBusy(true);
      setRunError(null);
      try {
        const [pred, rp] = await Promise.all([api.predict(body), api.regionPlan(body)]);
        setResult(pred);
        setPlan(rp);
      } catch (e) {
        setRunError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const submit = () => {
    if (!productId) return;
    void run({
      product_id: productId,
      discount_pct: discount,
      quantity,
      channel: SALES_CHANNELS.includes(campaignChannel ?? "") ? (campaignChannel as string) : "Web",
      customer_segment: segment,
      month,
      region: region === BEST ? null : region,
    });
  };

  if (error) return <Loading error={error}>{null}</Loading>;
  if (!options) return <LoadingText />;

  const categoryCard = catCards.find((c) => c.category === category) ?? null;
  const campaignCard = campCards.find((c) => c.channel === campaignChannel) ?? null;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Venture deck" hint="pick a category">
          <CardDeckCarousel
            slides={ventureSlides}
            autoplay={0}
            height={360}
            background="transparent"
            selectedIndex={Math.max(0, catCards.findIndex((c) => c.category === category))}
            onSelect={(i) => setCategory(catCards[i]?.category ?? null)}
          />
        </Panel>
        <Panel title="Campaign deck" hint={campaignNote}>
          <CardDeckCarousel
            slides={campaignSlides}
            autoplay={0}
            height={360}
            background="transparent"
            selectedIndex={Math.max(0, campCards.findIndex((c) => c.channel === campaignChannel))}
            onSelect={(i) => {
              const ch = campCards[i]?.channel ?? null;
              setCampaignChannel(ch);
            }}
          />
        </Panel>
      </div>

      <Panel title="Proposed sale" hint="region is never an input">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Product
            <select
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
              className="rounded-[var(--radius)] border border-border bg-[rgba(20,20,20,0.75)] px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60"
            >
              {products.map((p) => (
                <option key={p.product_id} value={p.product_id}>
                  {p.product_id} - {p.category}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Discount {(discount * 100).toFixed(0)}%
            <input
              type="range"
              min={0}
              max={25}
              step={1}
              value={Math.round(discount * 100)}
              onChange={(e) => setDiscount(Number(e.target.value) / 100)}
              className="mt-2 w-full"
            />
          </label>

          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Quantity
            <input
              type="number"
              min={1}
              max={5}
              value={quantity}
              onChange={(e) => setQuantity(Math.min(5, Math.max(1, Number(e.target.value))))}
              className="rounded-[var(--radius)] border border-border bg-[rgba(20,20,20,0.75)] px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60"
            />
          </label>

          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Customer segment
            <select
              value={segment}
              onChange={(e) => setSegment(e.target.value)}
              className="rounded-[var(--radius)] border border-border bg-[rgba(20,20,20,0.75)] px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60"
            >
              {options.customer_segments.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Month
            <select
              value={month}
              onChange={(e) => setMonth(Number(e.target.value))}
              className="rounded-[var(--radius)] border border-border bg-[rgba(20,20,20,0.75)] px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60"
            >
              {options.months.map((mm) => (
                <option key={mm} value={mm}>
                  {mm}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Region
            <select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="rounded-[var(--radius)] border border-border bg-[rgba(20,20,20,0.75)] px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60"
            >
              <option value={BEST}>{BEST}</option>
              {options.regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={submit}
            disabled={busy || !productId}
            className="flex items-center gap-2 rounded-[var(--radius)] border border-primary/50 bg-primary px-4 py-2 text-sm font-medium text-[#141414] disabled:opacity-60"
          >
            <Sparkles className="h-3.5 w-3.5" />
            {busy ? "Checking" : "Check this sale"}
          </button>
          {categoryCard ? (
            <span className="text-[11px] text-muted-foreground">
              {categoryCard.category}: margin {categoryCard.margin_pct?.toFixed(1) ?? "n/a"}%,{" "}
              {pct(categoryCard.share_products_below_cost)} of products below cost
            </span>
          ) : null}
          {campaignCard ? (
            <span className="text-[11px] text-muted-foreground">
              {campaignCard.channel}: CPA {money(campaignCard.median_cpa)}, CVR{" "}
              {campaignCard.median_cvr === null ? "n/a" : pct(campaignCard.median_cvr)}
            </span>
          ) : null}
        </div>
        {runError ? <p className="mt-2 text-xs text-destructive">{runError}</p> : null}
      </Panel>

      {result && plan ? (
        <>
          <GlowCard customSize className="w-full" glowColor="blue">
            <div className="grid gap-4 p-4 sm:grid-cols-2 lg:grid-cols-4">
              <div>
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  Predicted profit
                </p>
                <p className="mt-1 text-2xl font-semibold tabular-nums text-foreground">
                  {money(result.profit.xgb)}
                </p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  linear {money(result.profit.linear)} | tree {money(result.profit.tree)}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Loss risk</p>
                <p className="mt-1 text-sm font-medium">
                  {result.loss_flag ? (
                    <span className="text-destructive">Predicted loss</span>
                  ) : (
                    <span className="text-primary">Profitable</span>
                  )}
                </p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  break-even discount {pct(result.break_even_discount)}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  History check
                </p>
                <p className="mt-1 text-sm font-medium text-foreground">{result.comparison.verdict}</p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  per unit {money(result.comparison.predicted_profit_per_unit)} vs historical{" "}
                  {money(result.comparison.historical_profit_per_unit)}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Action</p>
                <p className="mt-1 text-xs leading-snug text-foreground">{result.action}</p>
              </div>
            </div>
          </GlowCard>

          <GlowCard customSize className="w-full" glowColor="blue">
            <div className="p-4">
              <h2 className="mb-1 text-sm font-medium text-foreground">
                Region plan -{" "}
                {plan.mode === "best" ? "ranked options" : `discount ladder for ${plan.region?.region}`}
              </h2>
              <p className="mb-3 text-[11px] text-muted-foreground">
                {plan.basis ?? plan.suggested_price_note ?? plan.profit_note}
              </p>

              {plan.mode === "best" ? (
                <>
                  <div className="flex flex-col gap-1">
                    {(plan.regions ?? []).map((r) => (
                      <div
                        key={r.region}
                        className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3"
                      >
                        <span className="w-16 text-xs text-muted-foreground">{r.region}</span>
                        <BarRow
                          label={`completion ${pct(r.completion_rate)} | ${money(r.sales)} sales | loss share ${r.loss_sale_share === null ? "n/a" : pct(r.loss_sale_share)}`}
                          value={r.expected_realised_profit}
                          max={Math.max(
                            ...(plan.regions ?? []).map((x) => Math.abs(x.expected_realised_profit)),
                            0.0001,
                          )}
                          display={money(r.expected_realised_profit)}
                        />
                        <Pill tone={r.decision === "SELL" ? "sell" : "hold"}>{r.decision}</Pill>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-xs text-foreground">
                    Recommended region:{" "}
                    <span className="font-medium text-primary">{plan.recommended_region}</span>
                    {plan.kruskal_p !== null ? (
                      <span className="text-muted-foreground">
                        {" "}
                        (Kruskal p = {plan.kruskal_p.toFixed(2)})
                      </span>
                    ) : null}
                  </p>
                  {plan.tie_banner ? (
                    <p className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
                      <AlertTriangle className="h-3.5 w-3.5 text-primary" />
                      {plan.tie_banner}
                    </p>
                  ) : null}
                  <p className="mt-2 text-[11px] text-muted-foreground">{plan.profit_note}</p>
                </>
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="text-muted-foreground">
                        <tr>
                          <th className="py-2 pr-3 font-medium">Discount</th>
                          <th className="py-2 pr-3 font-medium">Net price</th>
                          <th className="py-2 pr-3 font-medium">Profit / unit</th>
                          <th className="py-2 pr-3 font-medium">XGB profit</th>
                          <th className="py-2 pr-3 font-medium">Expected realised</th>
                          <th className="py-2 font-medium">Decision</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(plan.ladder ?? []).map((row) => (
                          <tr key={row.discount_pct} className="border-t border-border">
                            <td className="py-2 pr-3 tabular-nums text-foreground">{row.discount_pct}%</td>
                            <td className="py-2 pr-3 tabular-nums text-muted-foreground">{money(row.net_price)}</td>
                            <td className="py-2 pr-3 tabular-nums text-muted-foreground">
                              {money(row.profit_per_unit)}
                            </td>
                            <td className="py-2 pr-3 tabular-nums text-muted-foreground">
                              {money(row.xgb_profit)}
                            </td>
                            <td className="py-2 pr-3 tabular-nums text-foreground">
                              {money(row.expected_realised_profit)}
                            </td>
                            <td className="py-2">
                              <Pill tone={row.decision === "SELL" ? "sell" : "hold"}>{row.decision}</Pill>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
                    <span>
                      max safe discount:{" "}
                      <span className="text-foreground">
                        {plan.max_safe_discount === undefined ? "n/a" : pct(plan.max_safe_discount)}
                      </span>
                    </span>
                    <span>
                      suggested price:{" "}
                      <span className="text-foreground">{money(plan.suggested_price)}</span>
                    </span>
                    <span>
                      break-even:{" "}
                      <span className="text-foreground">{pct(plan.break_even_discount)}</span>
                    </span>
                  </div>
                </>
              )}
            </div>
          </GlowCard>

          <div className="grid gap-4 lg:grid-cols-2">
            <GlowCard customSize className="w-full" glowColor="blue">
              <div className="p-4">
                <h2 className="text-sm font-medium text-foreground">Winner-region verdict</h2>
                <p className="mt-1 text-xs text-foreground">{result.region.verdict}</p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  accuracy {result.region.model_accuracy === null ? "n/a" : result.region.model_accuracy.toFixed(3)}{" "}
                  vs chance {result.region.chance === null ? "n/a" : result.region.chance.toFixed(2)} | signal{" "}
                  {result.region.signal} | Kruskal p{" "}
                  {result.region.kruskal_p === null ? "n/a" : result.region.kruskal_p.toFixed(2)}
                </p>
                <div className="mt-3">
                  {Object.entries(result.region.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([r, p]) => (
                      <BarRow key={r} label={r} value={p} max={1} display={pct(p)} />
                    ))}
                </div>
                {result.region.historical_leader ? (
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    {result.region.historical_leader.region} -{" "}
                    {result.region.historical_leader.label} (profit per sale{" "}
                    {money(result.region.historical_leader.profit_per_sale)})
                  </p>
                ) : null}
              </div>
            </GlowCard>

            <GlowCard customSize className="w-full" glowColor="blue">
              <div className="p-4">
                <h2 className="text-sm font-medium text-foreground">Why this prediction (SHAP)</h2>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  base {money(result.shap.base_value)} to prediction {money(result.shap.prediction)} (diff{" "}
                  {result.shap.base_plus_shap_error.toExponential(2)})
                </p>
                <div className="mt-3">
                  {result.shap.contributions.slice(0, 9).map((c) => (
                    <BarRow
                      key={c.feature}
                      label={`${c.feature} = ${text(c.value)}`}
                      value={c.shap}
                      max={Math.max(...result.shap.contributions.map((x) => Math.abs(x.shap)), 0.0001)}
                      display={money(c.shap)}
                      tone={c.shap >= 0 ? "up" : "down"}
                    />
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  red raises profit, blue lowers it
                </p>
              </div>
            </GlowCard>
          </div>

          <Panel title="History that supports this" hint="all numbers retrieved from the database">
            <div className="grid gap-4 lg:grid-cols-3">
              <div>
                <p className="text-xs text-muted-foreground">
                  completed sales {money(result.history.product.sales)}
                </p>
                <p className="text-xs text-muted-foreground">
                  mean profit per sale {money(result.history.product.mean_profit_per_sale)}
                </p>
                <p className="text-xs text-muted-foreground">
                  p10 / p50 / p90 {money(result.history.product.p10_profit_per_sale)} /{" "}
                  {money(result.history.product.p50_profit_per_sale)} /{" "}
                  {money(result.history.product.p90_profit_per_sale)}
                </p>
                <p className="text-xs text-muted-foreground">
                  loss-sale share {pct(result.history.product.loss_sale_share)} | cancel+return{" "}
                  {pct(result.history.product.return_cancel_rate)}
                </p>
                <p className="mt-2 text-xs text-foreground">
                  predicted per unit sits at the{" "}
                  {result.comparison.percentile_within_product_history === null
                    ? "n/a"
                    : `${result.comparison.percentile_within_product_history.toFixed(0)}th`}{" "}
                  percentile of this product's history
                </p>
              </div>
              <div>
                <p className="mb-1 text-xs text-muted-foreground">12-month mean profit trend</p>
                <Sparkline points={result.history.product_trend.map((t) => t.mean_profit ?? 0)} />
                <p className="mt-2 text-xs text-muted-foreground">
                  category margin {pct(result.history.category.margin)} | category Kruskal p{" "}
                  {result.history.kruskal_p_category === null
                    ? "n/a"
                    : result.history.kruskal_p_category.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="mb-1 text-xs text-muted-foreground">Category by region</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[11px]">
                    <thead className="text-muted-foreground">
                      <tr>
                        <th className="py-1 pr-2 font-medium">Region</th>
                        <th className="py-1 pr-2 font-medium">Profit/sale</th>
                        <th className="py-1 font-medium">Loss share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.history.regions.map((r, i) => (
                        <tr key={i} className="border-t border-border">
                          <td className="py-1 pr-2 text-foreground">{text(r.region)}</td>
                          <td className="py-1 pr-2 tabular-nums text-muted-foreground">
                            {money(Number(r.profit_per_sale))}
                          </td>
                          <td className="py-1 tabular-nums text-muted-foreground">
                            {pct(Number(r.loss_sale_share))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Similar past sales" hint="nearest discount then quantity">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    {(result.similar_sales[0] ? Object.keys(result.similar_sales[0]) : []).map((c) => (
                      <th key={c} className="py-2 pr-3 font-medium">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.similar_sales.map((row, i) => (
                    <tr key={i} className="border-t border-border">
                      {Object.keys(result.similar_sales[0] ?? {}).map((c) => (
                        <td key={c} className="py-2 pr-3 tabular-nums text-foreground">
                          {text(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Reasoning" hint="templated from retrieved numbers only">
            <p className="text-sm leading-relaxed text-foreground">{result.reasoning}</p>
          </Panel>
        </>
      ) : null}
    </div>
  );
}
