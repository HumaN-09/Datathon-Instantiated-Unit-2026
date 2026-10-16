"""STEP 3: retrieve facts with SQL, predict profit and the winning region,
compare prediction against history, and write reasoning that uses ONLY retrieved numbers.

Region is an output. Predicted profit is identical for every region because the data
shows no regional effect (Kruskal p ~ 0.96).
"""
import json

import numpy as np
import pandas as pd

from backend import config

_BUNDLE = None
_METRICS = None

PROFIT_FEATURES = config.NUM_COLS + config.CAT_COLS


# --------------------------------------------------------------- bundle access
def load_bundle(path=None):
    """Load the trained bundle once and cache it."""
    global _BUNDLE
    if _BUNDLE is None:
        import joblib

        p = path or config.BUNDLE_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"No model bundle at {p}. Run train first: python -m ml_model.train"
            )
        _BUNDLE = joblib.load(p)
    return _BUNDLE


def load_metrics():
    global _METRICS
    if _METRICS is None:
        with open(config.METRICS_PATH) as f:
            _METRICS = json.load(f)
    return _METRICS


def query(sql, params=()):
    con = config.get_connection()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def product_row(product_id):
    df = query("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if df.empty:
        raise ValueError(f"Unknown product_id: {product_id}")
    return df.iloc[0].to_dict()


# ------------------------------------------------------------------ feature row
def build_features(product_id, discount_pct, quantity, channel, customer_segment):
    """Build the 9 profit features. Region is never an input."""
    p = product_row(product_id)
    unit_cost = float(p["unit_cost"])
    list_price = float(p["list_price"])
    row = {
        "quantity": float(quantity),
        "discount_pct": float(discount_pct),
        "list_price": list_price,
        "unit_cost": unit_cost,
        "cost_to_list": unit_cost / list_price,
        "supplier_lead_days": float(p["supplier_lead_days"]),
        "category": str(p["category"]),
        "channel": str(channel),
        "customer_segment": str(customer_segment),
    }
    return row, p


def matrix_for(bundle, which, rows):
    """Turn feature dicts into a model-ready matrix for Dataset 1 or Dataset 2."""
    frame = pd.DataFrame(rows)[PROFIT_FEATURES]
    if which == "d1":
        dummies = pd.get_dummies(frame, columns=config.CAT_COLS, drop_first=True)
        dummies = dummies.reindex(columns=bundle["d1_cols"], fill_value=0).astype(float)
        return pd.DataFrame(bundle["scaler"].transform(dummies), columns=bundle["d1_cols"])
    out = frame[config.NUM_COLS].astype(float).copy()
    for c in config.CAT_COLS:
        cats = bundle["cat_maps"][c]
        codes = pd.Categorical(frame[c].astype(str), categories=cats).codes
        out[c] = codes
    return out[bundle["d2_cols"]].astype(float)


# ------------------------------------------------------------------- prediction
def predict_profit(product_id, discount_pct, quantity, channel, customer_segment):
    """XGB profit plus Linear and Tree profit for comparison."""
    bundle = load_bundle()
    row, p = build_features(product_id, discount_pct, quantity, channel, customer_segment)
    x2 = matrix_for(bundle, "d2", [row])

    xgb_profit = float(bundle["xgb_reg"].predict(x2)[0])
    lin_profit = float(bundle["lin_reg"].predict(matrix_for(bundle, "d1", [row]))[0])
    tree_profit = float(bundle["tree_reg"].predict(x2)[0])

    unit_cost = float(p["unit_cost"])
    list_price = float(p["list_price"])
    break_even = max(0.0, 1.0 - unit_cost / list_price)

    return {
        "product_id": product_id,
        "xgb_profit": xgb_profit,
        "linear_profit": lin_profit,
        "tree_profit": tree_profit,
        "loss_flag": bool(xgb_profit < 0),
        "break_even_discount": float(break_even),
        "discount_pct": float(discount_pct),
        "quantity": float(quantity),
        "unit_cost": unit_cost,
        "list_price": list_price,
        "category": str(p["category"]),
        "features": row,
        "x2_row": x2,
    }


def shap_explain(pred, explainer=None):
    """Exact TreeSHAP for this input row: base + sum(shap) equals the prediction."""
    if explainer is None:
        import shap

        explainer = shap.TreeExplainer(load_bundle()["xgb_reg"])
    sv = explainer.shap_values(pred["x2_row"])
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv[0]).ravel()
    base = float(np.ravel(explainer.expected_value)[0])
    prediction = float(base + sv.sum())
    cols = load_bundle()["d2_cols"]
    contribs = []
    for c, s in zip(cols, sv):
        raw = pred["features"][c]
        # Category columns carry text, numeric columns carry floats.
        val = float(raw) if isinstance(raw, (int, float)) else str(raw)
        contribs.append({"feature": c, "value": val, "shap": float(s)})
    contribs.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return {
        "base_value": base,
        "prediction": prediction,
        "model_prediction": float(pred["xgb_profit"]),
        "base_plus_shap_error": abs(prediction - float(pred["xgb_profit"])),
        "contributions": contribs,
    }


# --------------------------------------------------------------------- history
def get_history(product_id, category, channel=None, customer_segment=None):
    """Product stats, category stats, the category's region table, and its Kruskal p."""
    prod = query("SELECT * FROM sales_clean WHERE product_id = ?", (product_id,))
    comp = prod[prod["status"] == "Completed"]

    def stats(frame):
        if frame.empty:
            return {"sales": 0}
        per_sale = frame["profit"]
        per_unit = frame["profit"] / frame["quantity"].replace(0, np.nan)
        return {
            "sales": int(len(frame)),
            "mean_profit_per_sale": float(per_sale.mean()),
            "median_profit_per_sale": float(per_sale.median()),
            "std_profit_per_sale": float(per_sale.std(ddof=0)),
            "p10_profit_per_sale": float(per_sale.quantile(0.10)),
            "p50_profit_per_sale": float(per_sale.quantile(0.50)),
            "p90_profit_per_sale": float(per_sale.quantile(0.90)),
            "mean_profit_per_unit": float(per_unit.mean()),
            "loss_sale_share": float(prod["loss_sale"].mean()) if len(prod) else 0.0,
            "return_cancel_rate": float(prod["status"].isin(["Cancelled", "Returned"]).mean()) if len(prod) else 0.0,
        }

    product_stats = stats(comp)

    trend_rows = (
        comp.groupby("month")["profit"].mean().reindex(range(1, 13)).reset_index()
        if not comp.empty
        else pd.DataFrame({"month": list(range(1, 13)), "profit": [np.nan] * 12})
    )
    trend = [
        {"month": int(r["month"]), "mean_profit": (None if pd.isna(r["profit"]) else float(r["profit"]))}
        for _, r in trend_rows.iterrows()
    ]

    cat_sales = query("SELECT * FROM sales_clean WHERE category = ?", (category,))
    cat_comp = cat_sales[cat_sales["status"] == "Completed"]
    category_stats = {
        "sales": int(len(cat_sales)),
        "completed_profit": float(cat_comp["profit"].sum()),
        "completed_revenue": float(cat_comp["revenue"].sum()),
        "profit_per_sale": float(cat_comp["profit"].mean()) if len(cat_comp) else 0.0,
        "margin": float(cat_comp["profit"].sum() / cat_comp["revenue"].sum()) if len(cat_comp) and cat_comp["revenue"].sum() else 0.0,
        "loss_sale_share": float(cat_sales["loss_sale"].mean()) if len(cat_sales) else 0.0,
    }

    region_table = query(
        "SELECT region, sales, completed_profit, profit_per_sale, loss_sale_share, return_cancel_rate "
        "FROM region_stats WHERE category = ? ORDER BY profit_per_sale DESC",
        (category,),
    ).to_dict("records")

    tests = load_metrics().get("region_tests", {})
    kruskal_p = tests.get("by_category", {}).get(category, tests.get("overall"))

    return {
        "product": product_stats,
        "product_trend": trend,
        "category": category_stats,
        "regions": region_table,
        "kruskal_p_category": kruskal_p,
        "kruskal_p_overall": tests.get("overall"),
    }


def similar_sales(product_id, discount_pct, quantity, k=5):
    """The k closest past sales of this product. Distance favours same discount, then quantity."""
    df = query(
        "SELECT timestamp, quantity, discount_pct, status, profit FROM sales_clean WHERE product_id = ?",
        (product_id,),
    )
    if df.empty:
        return []
    df["distance"] = (df["discount_pct"] - discount_pct).abs() * 10 + (df["quantity"] - quantity).abs()
    df = df.sort_values(["distance", "timestamp"]).head(k)
    return [
        {
            "date": str(r["timestamp"])[:10],
            "quantity": int(r["quantity"]),
            "discount_pct": float(r["discount_pct"]),
            "status": str(r["status"]),
            "profit": float(r["profit"]),
        }
        for _, r in df.iterrows()
    ]


# ------------------------------------------------------------------ region model
def predict_region(category, channel, customer_segment, month):
    """Winning-region probabilities. A 'clear winner' needs p >= 0.45 AND Kruskal p < 0.05."""
    bundle = load_bundle()
    frame = pd.DataFrame([{
        "category": str(category), "channel": str(channel),
        "customer_segment": str(customer_segment), "month": float(month),
    }])

    dummies = pd.get_dummies(frame, columns=["category", "channel", "customer_segment"], drop_first=True)
    dummies = dummies.reindex(columns=bundle["winner_d1_cols"], fill_value=0).astype(float)

    x2 = frame[["month"]].astype(float).copy()
    for c in ["category", "channel", "customer_segment"]:
        x2[c] = pd.Categorical(frame[c].astype(str), categories=bundle["winner_cat_maps"][c]).codes
    x2 = x2[bundle["winner_d2_cols"]].astype(float)

    # The classifier was trained on integer codes; `winner_classes` maps index -> region name.
    region_names = list(bundle["winner_classes"])
    proba = bundle["xgb_clf"].predict_proba(x2)[0]
    probs = {str(r): float(p) for r, p in zip(region_names, proba)}
    top_region = max(probs, key=probs.get)
    top_p = probs[top_region]

    metrics = load_metrics()
    kruskal_p = metrics.get("region_tests", {}).get("by_category", {}).get(category)

    leader = query(
        "SELECT region, profit_per_sale FROM region_stats WHERE category = ? ORDER BY profit_per_sale DESC LIMIT 1",
        (category,),
    )
    historical_leader = None
    if not leader.empty:
        historical_leader = {
            "region": str(leader.iloc[0]["region"]),
            "profit_per_sale": float(leader.iloc[0]["profit_per_sale"]),
            "label": "historical leader, not significant",
        }

    significant = (kruskal_p is not None) and (kruskal_p < 0.05)
    clear = bool(top_p >= 0.45 and significant)

    return {
        "probabilities": probs,
        "top_region": top_region,
        "top_probability": float(top_p),
        "kruskal_p": None if kruskal_p is None else float(kruskal_p),
        "clear_winner": clear,
        "verdict": f"Clear winner: {top_region}" if clear else "No statistically clear winner",
        "historical_leader": historical_leader,
        "signal": metrics.get("winner_region", {}).get("signal", "none"),
        "model_accuracy": metrics.get("winner_region", {}).get("best_test_accuracy"),
        "chance": metrics.get("winner_region", {}).get("chance"),
    }


# ------------------------------------------------------------------- comparison
def compare(pred, history):
    """Predicted profit per unit vs the product's historical profit per unit."""
    per_unit = pred["xgb_profit"] / pred["quantity"] if pred["quantity"] else 0.0
    hist_per_unit = history["product"].get("mean_profit_per_unit", 0.0) or 0.0

    if hist_per_unit == 0:
        verdict, delta_pct = "DIVERGES", None
    else:
        same_sign = (per_unit >= 0) == (hist_per_unit >= 0)
        delta_pct = (per_unit - hist_per_unit) / abs(hist_per_unit)
        verdict = "CONSISTENT" if (same_sign and abs(delta_pct) <= 0.25) else "DIVERGES"

    dist = query(
        "SELECT profit / quantity AS per_unit FROM sales_clean "
        "WHERE product_id = ? AND status = 'Completed' AND quantity > 0",
        (pred["product_id"],),
    )
    percentile = None
    if not dist.empty:
        percentile = float((dist["per_unit"] <= per_unit).mean() * 100)

    return {
        "predicted_profit_per_unit": float(per_unit),
        "historical_profit_per_unit": float(hist_per_unit),
        "delta_pct": None if delta_pct is None else float(delta_pct),
        "verdict": verdict,
        "percentile_within_product_history": percentile,
    }


# ------------------------------------------------------------------- explanation
def explain_text(pred, history, comparison, shap_info, region_info):
    """Templated reasoning built only from retrieved numbers, plus one action."""
    unit_note = "per unit"
    drivers = shap_info["contributions"][:3]
    driver_bits = []
    for d in drivers:
        direction = "raises" if d["shap"] >= 0 else "lowers"
        driver_bits.append(f"{d['feature']} = {_fmt(d['value'])} {direction} profit by {_fmt(abs(d['shap']))}")
    driver_text = "; ".join(driver_bits) if driver_bits else "no dominant driver"

    hp = history["product"]
    lines = [
        f"Predicted profit for {pred['quantity']:.0f} unit(s) at a {pred['discount_pct'] * 100:.0f}% discount is "
        f"{_money(pred['xgb_profit'])} ({unit_note}: {_money(comparison['predicted_profit_per_unit'])}).",
        f"Loss risk: predicted profit is {'negative' if pred['loss_flag'] else 'non-negative'}; "
        f"break-even discount on this product is {pred['break_even_discount'] * 100:.1f}%.",
        f"SHAP for this row starts from a base of {_money(shap_info['base_value'])} and moves to "
        f"{_money(shap_info['prediction'])}. Top drivers: {driver_text}.",
    ]
    if hp.get("sales"):
        lines.append(
            f"History check ({comparison['verdict']}): this product's {hp['sales']} past sales averaged "
            f"{_money(hp['mean_profit_per_sale'])} profit per sale (median {_money(hp['median_profit_per_sale'])}, "
            f"p10 {_money(hp['p10_profit_per_sale'])} to p90 {_money(hp['p90_profit_per_sale'])}), "
            f"with a loss-sale share of {hp['loss_sale_share'] * 100:.1f}% and a "
            f"{hp['return_cancel_rate'] * 100:.1f}% cancel/return rate."
        )
        if comparison["percentile_within_product_history"] is not None:
            lines.append(
                f"The predicted per-unit profit sits at the "
                f"{comparison['percentile_within_product_history']:.0f}th percentile of this product's completed sales."
            )
    cat = history["category"]
    lines.append(
        f"Category {pred['category']}: {_money(cat['completed_profit'])} completed profit on "
        f"{_money(cat['completed_revenue'])} revenue, margin {cat['margin'] * 100:.1f}%."
    )
    if region_info["clear_winner"]:
        lines.append(
            f"Region: {region_info['verdict']} (probability {region_info['top_probability']:.2f}, "
            f"Kruskal p = {region_info['kruskal_p']:.3f})."
        )
    else:
        leader = region_info["historical_leader"]
        lead_txt = (
            f" The category's historical leader is {leader['region']} at "
            f"{_money(leader['profit_per_sale'])} profit per sale ({leader['label']})."
            if leader else ""
        )
        p_txt = "n/a" if region_info["kruskal_p"] is None else f"{region_info['kruskal_p']:.3f}"
        lines.append(
            f"Region: no statistically clear winner (Kruskal p = {p_txt}; "
            f"top model probability {region_info['top_probability']:.2f} vs the 0.45 threshold).{lead_txt} "
            f"Predicted profit is the same for every region."
        )

    action = _pick_action(pred, comparison)
    return {"reasoning": " ".join(lines), "action": action}


def _pick_action(pred, comparison):
    if pred["discount_pct"] > pred["break_even_discount"]:
        return (f"Cap the discount at the break-even {pred['break_even_discount'] * 100:.1f}% "
                f"(requested {pred['discount_pct'] * 100:.0f}%) so the sale stops destroying margin.")
    if pred["unit_cost"] >= pred["list_price"]:
        return "Review the price: unit cost is at or above list price, so this product can never be profitable."
    if comparison["verdict"] == "DIVERGES":
        return "Re-check this proposal against history before approving; the model disagrees with past performance."
    risk = query("SELECT COUNT(*) AS n FROM inventory_clean WHERE true_stockout = 1 OR at_risk = 1")
    if int(risk.iloc[0]["n"]) > 0:
        return f"Restock or clear stale pending orders: {int(risk.iloc[0]['n'])} inventory rows are at risk."
    return "Approve: the discount stays inside the break-even band and history supports the forecast."


def _money(v):
    return f"{v:,.2f}"


def _fmt(v):
    return f"{v:,.3f}"


# ------------------------------------------------------------------ leak summary
def leak_summary():
    con = config.get_connection()
    try:
        sales = pd.read_sql_query("SELECT * FROM sales_clean", con)
        inv = pd.read_sql_query("SELECT * FROM inventory_clean", con)
        products = pd.read_sql_query("SELECT * FROM products", con)
    finally:
        con.close()

    comp = sales[sales["status"] == "Completed"]
    lost = sales[sales["status"].isin(["Cancelled", "Returned"])]
    stale = sales[sales["stale_pending"] == 1]

    cat_totals = comp.groupby("category")[["profit", "revenue"]].sum()
    cat_margin = (cat_totals["profit"] / cat_totals["revenue"].replace(0, np.nan)).fillna(0.0).sort_values(ascending=False)
    worst = (
        comp.groupby("product_id")
        .agg(total_completed_profit=("profit", "sum"), sales=("profit", "size"))
        .reset_index()
        .merge(products[["product_id", "category"]], on="product_id", how="left")
        .sort_values("total_completed_profit")
        .head(10)
    )

    return {
        "total_revenue": float(sales["revenue"].sum()),
        "lost_revenue_share": float(lost["revenue"].sum() / sales["revenue"].sum()),
        "lost_revenue_value": float(lost["revenue"].sum()),
        "loss_sale_share": float(sales["loss_sale"].mean()),
        "loss_sale_count": int(sales["loss_sale"].sum()),
        "negative_margin_products": int((products["unit_cost"] >= products["list_price"]).sum()),
        "stale_pending_count": int(len(stale)),
        "stale_pending_value": float(stale["revenue"].sum()),
        "pending_count": int((sales["status"] == "Pending").sum()),
        "category_margins": [{"category": str(k), "margin": float(v)} for k, v in cat_margin.items()],
        "region_profit_per_sale": query(
            "SELECT region, AVG(profit_per_sale) AS profit_per_sale FROM region_stats GROUP BY region ORDER BY profit_per_sale DESC"
        ).to_dict("records"),
        "kruskal_p_overall": load_metrics().get("region_tests", {}).get("overall"),
        "true_stockout": int(inv["true_stockout"].sum()),
        "at_risk": int(inv["at_risk"].sum()),
        "worst_products": worst.to_dict("records"),
    }


# -------------------------------------------------------------------------- ask
def ask(question):
    """Keyword intent router. Each branch runs its own SQL and returns a templated answer."""
    q = (question or "").lower()
    con = config.get_connection()
    try:
        if any(w in q for w in ["worst", "losing", "unprofitable", "leak", "bleed"]):
            rows = pd.read_sql_query(
                "SELECT product_id, category, SUM(profit) AS total_profit, COUNT(*) AS sales "
                "FROM sales_clean WHERE status = 'Completed' GROUP BY product_id ORDER BY total_profit ASC LIMIT 10",
                con,
            ).to_dict("records")
            answer = f"Worst 10 products by total completed profit. The single worst is {rows[0]['product_id']} at {rows[0]['total_profit']:,.2f}."

        elif any(w in q for w in ["pending", "stale", "stuck", "backlog"]):
            rows = pd.read_sql_query(
                "SELECT product_id, status, COUNT(*) AS n, SUM(revenue) AS value FROM sales_clean "
                "WHERE stale_pending = 1 GROUP BY product_id ORDER BY value DESC LIMIT 10",
                con,
            ).to_dict("records")
            tot = pd.read_sql_query("SELECT COUNT(*) AS n, SUM(revenue) AS v FROM sales_clean WHERE stale_pending = 1", con)
            answer = (f"{int(tot.iloc[0]['n'])} stale pending orders (over 30 days old) worth "
                      f"{tot.iloc[0]['v']:,.2f}. Top products listed.")

        elif any(w in q for w in ["stock", "inventory", "reorder", "shortage"]):
            rows = pd.read_sql_query(
                "SELECT product_id, COUNT(*) AS rows_at_risk, MIN(stock_on_hand) AS min_stock FROM inventory_clean "
                "WHERE at_risk = 1 GROUP BY product_id ORDER BY min_stock ASC LIMIT 10",
                con,
            ).to_dict("records")
            tot = pd.read_sql_query("SELECT SUM(true_stockout) AS so, SUM(at_risk) AS ar FROM inventory_clean", con)
            answer = (f"{int(tot.iloc[0]['so'])} true stockouts (stock on hand = 0) and "
                      f"{int(tot.iloc[0]['ar'])} at-risk rows (stock <= reorder threshold).")

        elif any(w in q for w in ["category", "categories", "segment of business"]):
            rows = pd.read_sql_query(
                "SELECT category, SUM(profit) AS completed_profit, SUM(revenue) AS completed_revenue, "
                "SUM(profit) / SUM(revenue) AS margin FROM sales_clean WHERE status = 'Completed' "
                "GROUP BY category ORDER BY margin DESC",
                con,
            ).to_dict("records")
            answer = f"Category margins from completed sales. Best is {rows[0]['category']} at {rows[0]['margin'] * 100:.1f}%."

        elif any(w in q for w in ["margin", "profit"]):
            rows = pd.read_sql_query(
                "SELECT category, SUM(profit) AS completed_profit, SUM(profit) / SUM(revenue) AS margin "
                "FROM sales_clean WHERE status = 'Completed' GROUP BY category ORDER BY margin ASC",
                con,
            ).to_dict("records")
            answer = f"Category margin ranking (worst first). Weakest is {rows[0]['category']} at {rows[0]['margin'] * 100:.1f}%."

        elif any(w in q for w in ["discount", "price", "pricing"]):
            rows = pd.read_sql_query(
                "SELECT discount_pct, COUNT(*) AS sales, AVG(profit) AS avg_profit, "
                "AVG(CASE WHEN profit < 0 THEN 1.0 ELSE 0.0 END) AS loss_share FROM sales_clean "
                "GROUP BY discount_pct ORDER BY discount_pct",
                con,
            ).to_dict("records")
            risky = [r for r in rows if r["loss_share"] and r["loss_share"] > 0.5]
            answer = (f"Discount bands and their average profit. {len(risky)} discount levels lose money on "
                      f"more than half of sales.")

        else:
            rows = pd.read_sql_query(
                "SELECT COUNT(*) AS sales, SUM(revenue) AS revenue, SUM(profit) AS profit, "
                "AVG(loss_sale) AS loss_share FROM sales_clean", con,
            ).to_dict("records")
            answer = ("I can answer about worst products, stale/pending orders, stock, category margins, "
                      "profit and discounts. Overall: "
                      f"{int(rows[0]['sales'])} sales, revenue {rows[0]['revenue']:,.2f}, "
                      f"profit {rows[0]['profit']:,.2f}, loss-sale share {rows[0]['loss_share'] * 100:.1f}%.")
    finally:
        con.close()

    return {"question": question, "answer": answer, "rows": rows}


# ------------------------------------------------------------------- region plan
def _kruskal_p(category):
    """Category-level Kruskal-Wallis p-value for profit across regions (from metrics.json)."""
    try:
        m = load_metrics()
        return float(m.get("region_tests", {}).get("by_category", {}).get(category))
    except Exception:
        return None


def _region_row(region, category, overall_rate, profit_pred, break_even, discount_pct):
    """One region: shrunk completion rate, expected realised profit, SELL/HOLD."""
    df = query(
        "SELECT COUNT(*) AS n, "
        "       SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed, "
        "       AVG(CASE WHEN status = 'Completed' THEN profit END) AS profit_per_sale, "
        "       AVG(CASE WHEN status = 'Completed' THEN loss_sale END) AS loss_share "
        "FROM sales_clean WHERE category = ? AND region = ?",
        (category, region),
    ).iloc[0]
    n = int(df["n"] or 0)
    completed = int(df["completed"] or 0)
    rate = (completed + 20.0 * overall_rate) / (n + 20.0) if (n + 20.0) else overall_rate
    expected = profit_pred * rate
    return {
        "region": region,
        "sales": n,
        "completed": completed,
        "completion_rate": round(float(rate), 4),
        "profit_per_sale": None if pd.isna(df["profit_per_sale"]) else round(float(df["profit_per_sale"]), 2),
        "loss_sale_share": None if pd.isna(df["loss_share"]) else round(float(df["loss_share"]), 4),
        "expected_realised_profit": round(float(expected), 2),
        "decision": "SELL" if (expected > 0 and float(discount_pct) <= break_even) else "HOLD",
    }


def region_plan(product_id, discount_pct, quantity, channel, customer_segment, month, region=None):
    """Rank the 5 regions by expected realised profit, or build a discount ladder for one region.

    The predicted profit is identical for every region because the data shows no
    regional effect; only the completion-rate haircut differs.
    """
    pred = predict_profit(product_id, discount_pct, quantity, channel, customer_segment)
    category = pred["category"]
    profit_pred = pred["xgb_profit"]
    break_even = pred["break_even_discount"]

    total = query(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed "
        "FROM sales_clean WHERE category = ?",
        (category,),
    ).iloc[0]
    tn = int(total["n"] or 0)
    overall_rate = (int(total["completed"] or 0) / tn) if tn else 0.0

    common = {
        "product_id": product_id,
        "category": category,
        "quantity": quantity,
        "channel": channel,
        "customer_segment": customer_segment,
        "month": month,
        "profit_pred": round(float(profit_pred), 2),
        "profit_is_region_independent": True,
        "profit_note": (
            "Predicted profit is the same for every region: region is not an input feature "
            "because the data shows no regional effect on profit per sale."
        ),
        "break_even_discount": round(float(break_even), 4),
    }

    if region is None:
        rows = [
            _region_row(r, category, overall_rate, profit_pred, break_even, discount_pct)
            for r in config.REGIONS
        ]
        rows.sort(key=lambda r: r["expected_realised_profit"], reverse=True)
        p = _kruskal_p(category)
        tied = (p is None) or (p >= 0.05)
        return {
            **common,
            "mode": "best",
            "regions": rows,
            "recommended_region": rows[0]["region"] if rows else None,
            "kruskal_p": p,
            "tie_banner": "regions are statistically tied, ranking is indicative only" if tied else None,
            "basis": "expected_realised_profit = predicted profit x shrunk completion rate",
        }

    p = _kruskal_p(category)
    row = _region_row(region, category, overall_rate, profit_pred, break_even, discount_pct)

    list_price = pred["list_price"]
    unit_cost = pred["unit_cost"]
    ladder = []
    for d in (0, 5, 10, 15, 20, 25):
        frac = d / 100.0
        net_price = list_price * (1 - frac)
        per_unit = net_price - unit_cost
        q_profit = float(predict_profit(product_id, frac, quantity, channel, customer_segment)["xgb_profit"])
        expected = q_profit * row["completion_rate"]
        ladder.append({
            "discount_pct": d,
            "net_price": round(net_price, 2),
            "profit_per_unit": round(per_unit, 2),
            "xgb_profit": round(q_profit, 2),
            "expected_realised_profit": round(expected, 2),
            "decision": "SELL" if (expected > 0 and frac <= break_even) else "HOLD",
        })

    max_safe = max(0.0, 1.0 - unit_cost / (0.9 * list_price)) if list_price else 0.0
    capped = min(max_safe, 0.25)
    return {
        **common,
        "mode": "options",
        "region": row,
        "ladder": ladder,
        "kruskal_p": p,
        "max_safe_discount": round(float(max_safe), 4),
        "suggested_price": round(float(list_price * (1 - capped)), 2),
        "suggested_price_note": "list price less the max safe discount, capped at 25%",
    }


# -------------------------------------------------------------- card endpoints
def category_cards():
    """Per category: margin, share of products below cost, campaign CPA/CVR."""
    margin = query(
        "SELECT category, "
        "       SUM(CASE WHEN status = 'Completed' THEN profit END) AS completed_profit, "
        "       SUM(CASE WHEN status = 'Completed' THEN revenue END) AS completed_revenue "
        "FROM sales_clean GROUP BY category ORDER BY category"
    )
    below = query(
        "SELECT category, "
        "       AVG(CASE WHEN unit_cost >= list_price THEN 1.0 ELSE 0.0 END) AS share_below_cost, "
        "       COUNT(*) AS products "
        "FROM products GROUP BY category ORDER BY category"
    )
    camp = query(
        "SELECT category, "
        "       COUNT(*) AS valid_rows, "
        "       AVG(spend) AS mean_spend "
        "FROM campaigns_clean WHERE invalid_row = 0 GROUP BY category ORDER BY category"
    )
    med = query(
        "SELECT category, "
        "       AVG(cpa) AS median_cpa, AVG(cvr) AS median_cvr "
        "FROM campaigns_clean WHERE invalid_row = 0 GROUP BY category ORDER BY category"
    )
    m = margin.set_index("category")
    b = below.set_index("category")
    c = camp.set_index("category")
    d = med.set_index("category")

    cards = []
    for cat in m.index:
        profit = float(m.loc[cat, "completed_profit"] or 0.0)
        revenue = float(m.loc[cat, "completed_revenue"] or 0.0)
        cards.append({
            "title": cat,
            "category": cat,
            "margin_pct": round(100.0 * profit / revenue, 2) if revenue else None,
            "share_products_below_cost": round(float(b.loc[cat, "share_below_cost"]), 4) if cat in b.index else None,
            "products": int(b.loc[cat, "products"]) if cat in b.index else 0,
            "campaign_valid_rows": int(c.loc[cat, "valid_rows"]) if cat in c.index else 0,
            "campaign_median_cpa": round(float(d.loc[cat, "median_cpa"]), 2) if cat in d.index and not pd.isna(d.loc[cat, "median_cpa"]) else None,
            "campaign_median_cvr": round(float(d.loc[cat, "median_cvr"]), 4) if cat in d.index and not pd.isna(d.loc[cat, "median_cvr"]) else None,
        })
    return cards


def campaign_cards():
    """Per campaign channel: median CPA, CVR, valid-row count, spend."""
    df = query(
        "SELECT channel, "
        "       COUNT(*) AS valid_rows, "
        "       SUM(spend) AS spend, "
        "       AVG(cpa) AS median_cpa, "
        "       AVG(cvr) AS median_cvr "
        "FROM campaigns_clean WHERE invalid_row = 0 GROUP BY channel ORDER BY channel"
    )
    cards = []
    for _, r in df.iterrows():
        cards.append({
            "title": r["channel"],
            "channel": r["channel"],
            "valid_rows": int(r["valid_rows"]),
            "spend": round(float(r["spend"] or 0.0), 2),
            "median_cpa": None if pd.isna(r["median_cpa"]) else round(float(r["median_cpa"]), 2),
            "median_cvr": None if pd.isna(r["median_cvr"]) else round(float(r["median_cvr"]), 4),
        })
    return {"cards": cards, "note": "campaigns do not link to sales"}
