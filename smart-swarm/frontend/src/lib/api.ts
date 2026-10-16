export type Row = Record<string, string | number | null>;

export type Options = {
  products: { product_id: string; category: string; list_price: number; unit_cost: number }[];
  categories: string[];
  channels: string[];
  customer_segments: string[];
  months: number[];
  regions: string[];
  regions_are_output_only: boolean;
};

export type Summary = {
  total_revenue: number;
  lost_revenue_share: number;
  lost_revenue_value: number;
  loss_sale_share: number;
  loss_sale_count: number;
  negative_margin_products: number;
  stale_pending_count: number;
  stale_pending_value: number;
  pending_count: number;
  true_stockout: number;
  at_risk: number;
  category_margins: { category: string; margin: number }[];
  region_profit_per_sale: { region: string; profit_per_sale: number }[];
  kruskal_p_overall: number | null;
  worst_products: { product_id: string; category: string; total_completed_profit: number; sales: number }[];
};

export type CategoryCard = {
  title: string;
  category: string;
  margin_pct: number | null;
  share_products_below_cost: number | null;
  products: number;
  campaign_valid_rows: number;
  campaign_median_cpa: number | null;
  campaign_median_cvr: number | null;
};

export type CampaignCard = {
  title: string;
  channel: string;
  valid_rows: number;
  spend: number;
  median_cpa: number | null;
  median_cvr: number | null;
};

export type RegionRow = {
  region: string;
  sales: number;
  completed: number;
  completion_rate: number;
  profit_per_sale: number | null;
  loss_sale_share: number | null;
  expected_realised_profit: number;
  decision: "SELL" | "HOLD";
};

export type LadderRow = {
  discount_pct: number;
  net_price: number;
  profit_per_unit: number;
  xgb_profit: number;
  expected_realised_profit: number;
  decision: "SELL" | "HOLD";
};

export type RegionPlan = {
  mode: "best" | "options";
  product_id: string;
  category: string;
  profit_pred: number;
  profit_is_region_independent: boolean;
  profit_note: string;
  break_even_discount: number;
  kruskal_p: number | null;
  regions?: RegionRow[];
  recommended_region?: string | null;
  tie_banner?: string | null;
  basis?: string;
  region?: RegionRow;
  ladder?: LadderRow[];
  max_safe_discount?: number;
  suggested_price?: number;
  suggested_price_note?: string;
};

export type ProductStats = {
  sales: number;
  mean_profit_per_sale?: number;
  median_profit_per_sale?: number;
  std_profit_per_sale?: number;
  p10_profit_per_sale?: number;
  p50_profit_per_sale?: number;
  p90_profit_per_sale?: number;
  mean_profit_per_unit?: number;
  loss_sale_share?: number;
  return_cancel_rate?: number;
};

export type RegionVerdict = {
  probabilities: Record<string, number>;
  top_region: string;
  top_probability: number;
  kruskal_p: number | null;
  clear_winner: boolean;
  verdict: string;
  historical_leader: { region: string; profit_per_sale: number; label: string } | null;
  signal: string;
  model_accuracy: number | null;
  chance: number | null;
};

export type PredictResult = {
  product_id: string;
  category: string;
  profit: { xgb: number; linear: number; tree: number; per_unit_xgb: number };
  loss_flag: boolean;
  break_even_discount: number;
  region: RegionVerdict;
  history: {
    product: ProductStats;
    product_trend: { month: number; mean_profit: number | null }[];
    category: {
      sales: number;
      completed_profit: number;
      completed_revenue: number;
      profit_per_sale: number;
      margin: number;
      loss_sale_share: number;
    };
    regions: Row[];
    kruskal_p_category: number | null;
    kruskal_p_overall: number | null;
    latest_inventory: Row | null;
  };
  similar_sales: Row[];
  comparison: {
    predicted_profit_per_unit: number;
    historical_profit_per_unit: number;
    delta_pct: number | null;
    verdict: "CONSISTENT" | "DIVERGES";
    percentile_within_product_history: number | null;
  };
  shap: {
    base_value: number;
    prediction: number;
    model_prediction: number;
    base_plus_shap_error: number;
    contributions: { feature: string; value: number | string; shap: number }[];
  };
  reasoning: string;
  action: string;
};

export type Metrics = {
  dataset?: { train_rows: number; test_rows: number };
  model_registry?: {
    model: string;
    task: string;
    dataset: string;
    features: string;
    target: string;
    metric: string;
    role: string;
    r2_test?: number;
    mae_test?: number;
    accuracy?: number;
    log_loss?: number;
  }[];
  not_trained?: string;
  self_checks?: { check: string; result: "PASS" | "FAIL" }[];
  self_checks_all_pass?: boolean;
  profit_regression?: Record<string, Record<string, number>>;
  winner_region?: {
    models: Record<string, Record<string, number>>;
    classes: string[];
    chance: number;
    best_test_accuracy: number;
    majority_baseline: number;
    signal: string;
  };
  region_tests?: { overall: number; by_category: Record<string, number> };
  shap?: { base_value: number; max_base_plus_shap_error: number; sample_rows: number };
};

export type AskResult = { question: string; answer: string; rows: Row[] };

export type PlanRequest = {
  product_id: string;
  discount_pct: number;
  quantity: number;
  channel: string;
  customer_segment: string;
  month: number;
  region: string | null;
};

const JSON_HEADERS = { "Content-Type": "application/json" };

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  options: () => get<Options>("/api/options"),
  summary: () => get<Summary>("/api/summary"),
  categoryCards: () => get<{ cards: CategoryCard[] }>("/api/category-cards"),
  campaignCards: () => get<{ cards: CampaignCard[]; note: string }>("/api/campaign-cards"),
  metrics: () => get<Metrics>("/api/metrics"),
  predict: (body: PlanRequest) => post<PredictResult>("/api/predict", body),
  regionPlan: (body: PlanRequest) => post<RegionPlan>("/api/region-plan", body),
  ask: (question: string) => post<AskResult>("/api/ask", { question }),
};

export const CHART_GROUPS: { title: string; names: string[] }[] = [
  {
    title: "Profit regression",
    names: [
      "profit_hist",
      "corr_heatmap",
      "pred_vs_actual",
      "residual_hist",
      "model_comparison",
      "xgb_reg_training_curve",
      "overfit_check",
      "xgb_feature_importance",
      "linear_coefficients",
      "tree_plot",
    ],
  },
  {
    title: "Winning region",
    names: [
      "region_profit_box",
      "region_winner_share",
      "winner_vs_chance",
      "winner_confusion_matrix",
      "xgb_clf_training_curve",
    ],
  },
  { title: "SHAP", names: ["shap_summary", "shap_bar", "shap_winner_bar"] },
];

/**
 * One line per chart, taken from what the plotting code in ml_model/train.py
 * actually draws. Shown when a chart is opened in the evidence grid.
 */
export type ChartNote = { title: string; note: string };

export const CHART_NOTES: Record<string, ChartNote> = {
  profit_hist: {
    title: "Profit per sale",
    note: "Distribution of profit on every completed sale. The left tail is the money leak: sales that were made below cost.",
  },
  corr_heatmap: {
    title: "Correlation heatmap",
    note: "Correlation between quantity, discount, list price, unit cost, cost-to-list, lead time, revenue and profit.",
  },
  pred_vs_actual: {
    title: "Predicted vs actual profit",
    note: "All three regressors against the held-out months 10-12. Points hugging the dashed 45-degree line mean the what-if price is trustworthy.",
  },
  residual_hist: {
    title: "XGB residuals (test)",
    note: "Actual minus predicted profit for the production regressor. Centred on zero with a narrow spread.",
  },
  model_comparison: {
    title: "Test R2 and MAE",
    note: "LinearRegression, DecisionTreeRegressor and XGBRegressor side by side, with the linear model's MAE as the baseline every other model must beat.",
  },
  xgb_reg_training_curve: {
    title: "XGB regressor training curve",
    note: "RMSE per boosting round on train and test. The test curve stops improving where early stopping stopped the run, so rounds were not wasted on overfitting.",
  },
  overfit_check: {
    title: "Overfit check",
    note: "Train R2 against test R2 for each regressor. The pairs stay close together.",
  },
  xgb_feature_importance: {
    title: "XGB feature importance",
    note: "Gain-based importance of the nine features. Quantity, unit cost and discount carry the signal; region is absent by design.",
  },
  linear_coefficients: {
    title: "Linear coefficients (scaled)",
    note: "Scaled coefficients of the linear baseline, useful for reading the direction of each feature: more discount means less profit.",
  },
  tree_plot: {
    title: "Decision tree (first 3 levels)",
    note: "The top three levels of the DecisionTreeRegressor: a readable second opinion on the XGB numbers.",
  },
  region_profit_box: {
    title: "Profit per sale by region",
    note: "Completed profit per sale for each region. The title carries the Kruskal-Wallis p-value: the boxes overlap, so the regions are statistically tied.",
  },
  region_winner_share: {
    title: "Region winner share",
    note: "How often each region wins a category-month cell, against the 20% line that means equal chance.",
  },
  winner_vs_chance: {
    title: "Region accuracy vs majority vs chance",
    note: "The winning-region classifiers against the majority-class baseline and 1-in-5 chance. Accuracy lands at chance, which is why the app reports a tie instead of a confident winner.",
  },
  winner_confusion_matrix: {
    title: "Winner confusion matrix (XGB)",
    note: "Actual against predicted winning region on the held-out months. The diagonal is weak, the honest picture of a near-chance target.",
  },
  xgb_clf_training_curve: {
    title: "XGB winner-classifier training curve",
    note: "Multi-class log-loss per round for the region classifier.",
  },
  shap_summary: {
    title: "SHAP beeswarm - XGB profit regressor",
    note: "Every sampled sale as a dot: how far each feature pushed the prediction, and which way. This is what explains a single what-if answer in the Predict tab.",
  },
  shap_bar: {
    title: "Mean |SHAP| - XGB profit regressor",
    note: "The same explanation ranked by average impact, the feature ranking behind every prediction.",
  },
  shap_winner_bar: {
    title: "Mean |SHAP| - winner classifier",
    note: "Impact summed over the five region classes for a model that only matches chance. Read it as a warning, not a vote.",
  },
};

export function money(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

export function text(v: unknown): string {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(v);
}
