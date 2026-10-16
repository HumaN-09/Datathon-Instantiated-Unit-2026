"""STEP 1 / 1B: build money_leak.db from the raw CSVs and print verification checks.

Data rules (do not change):
  unit_price is ALREADY net of discount. revenue = quantity * unit_price.
  profit = revenue - quantity * unit_cost. Never apply discount again.
  Join transactions -> products on product_id, transactions -> customers on customer_id.
  Never join inventory or campaigns to sales.
"""
import json

import numpy as np
import pandas as pd

from backend import config


def read_raw():
    """Load the five raw CSVs, stopping if any of them is missing."""
    missing = config.check_raw_files()
    if missing:
        raise SystemExit(f"STOP: missing raw file(s): {', '.join(missing)}")
    return {name: pd.read_csv(path) for name, path in config.RAW_FILES.items()}


def build_sales_clean(tx, products, customers):
    """Merged, feature-engineered sales table."""
    df = tx.merge(products, on="product_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    df["month"] = df["timestamp"].dt.month
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["profit"] = df["revenue"] - df["quantity"] * df["unit_cost"]
    df["loss_sale"] = (df["profit"] < 0).astype(int)
    df["cost_to_list"] = df["unit_cost"] / df["list_price"]
    # unit_cost >= list_price means the product can never be sold at a profit.
    df["neg_margin_product"] = (df["unit_cost"] >= df["list_price"]).astype(int)
    # A sale timestamped before the customer signed up is a data-integrity leak.
    df["pre_signup"] = (df["timestamp"] < df["signup_date"]).astype(int)

    latest = df["timestamp"].max()
    df["age_days"] = (latest - df["timestamp"]).dt.days
    # Stale pending = still Pending and older than 30 days.
    df["stale_pending"] = (
        (df["status"] == "Pending") & (df["age_days"] > 30)
    ).astype(int)
    return df


def build_inventory_clean(inv):
    """Inventory plus reliability-safe stockout flags.

    inventory.stockout_flag is unreliable, so it is ignored on purpose.
    """
    df = inv.copy()
    df["true_stockout"] = (df["stock_on_hand"] == 0).astype(int)
    df["at_risk"] = (df["stock_on_hand"] <= df["reorder_threshold"]).astype(int)
    return df


def build_campaigns_clean(camp):
    """Campaigns plus validity flag and funnel metrics."""
    df = camp.copy()
    df["invalid_row"] = ((df["clicks"] > df["impressions"]) | (df["conversions"] > df["clicks"])).astype(int)
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
    df["cvr"] = df["conversions"] / df["clicks"].replace(0, np.nan)
    df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
    df["cpa"] = df["spend"] / df["conversions"].replace(0, np.nan)
    return df


def build_region_stats(sales):
    """STEP 1B: profit and risk by region x category."""
    rows = []
    for (region, category), g in sales.groupby(["region", "category"]):
        comp = g[g["status"] == "Completed"]
        completed_profit = float(comp["profit"].sum())
        completed_sales = int(len(comp))
        rows.append(
            {
                "region": region,
                "category": category,
                "sales": int(len(g)),
                "completed_profit": completed_profit,
                "completed_sales": completed_sales,
                "profit_per_sale": (completed_profit / completed_sales) if completed_sales else float("nan"),
                "loss_sale_share": float(g["loss_sale"].mean()),
                "return_cancel_rate": float(g["status"].isin(["Cancelled", "Returned"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_region_winners(sales):
    """STEP 1B: one row per (category, channel, customer_segment, month) cell.

    winner = the region with the highest completed profit inside that cell.
    """
    comp = sales[sales["status"] == "Completed"]
    if comp.empty:
        return pd.DataFrame(columns=["category", "channel", "customer_segment", "month", "winner", "winning_profit", "regions_in_cell"])

    grouped = (
        comp.groupby(["category", "channel", "customer_segment", "month", "region"])["profit"]
        .sum()
        .reset_index()
    )
    cells = (
        grouped.groupby(["category", "channel", "customer_segment", "month"])
        .agg(
            winner=("region", lambda s: s.iloc[0]),
            winning_profit=("profit", "max"),
            regions_in_cell=("region", "nunique"),
        )
        .reset_index()
    )
    # Re-pick the winner explicitly so the label always matches the max profit.
    best = grouped.sort_values("profit", ascending=False).drop_duplicates(
        ["category", "channel", "customer_segment", "month"]
    )
    cells["winner"] = cells.set_index(["category", "channel", "customer_segment", "month"]).index.map(
        best.set_index(["category", "channel", "customer_segment", "month"])["region"]
    )
    cells["winning_profit"] = cells.set_index(["category", "channel", "customer_segment", "month"]).index.map(
        best.set_index(["category", "channel", "customer_segment", "month"])["profit"]
    )
    return cells


def report(sales, products, inventory_clean, campaigns_clean, region_winners):
    """Print each check next to its expected value."""
    rows = []

    def chk(label, actual, expected):
        rows.append((label, actual, expected))

    comp = sales[sales["status"] == "Completed"]
    cr = sales[sales["status"].isin(["Cancelled", "Returned"])]

    chk("negative-margin products", int(products["neg_margin_product"].sum()) if "neg_margin_product" in products else int((products["unit_cost"] >= products["list_price"]).sum()), 76)
    chk("pre-signup sales", int(sales["pre_signup"].sum()), 797)
    chk("stale pending", f"{int(sales['stale_pending'].sum())} of {int((sales['status'] == 'Pending').sum())}", "824 of 918")
    chk("loss-making sales", f"{int(sales['loss_sale'].sum())} of {len(sales)} ({100 * sales['loss_sale'].mean():.1f}%)", "6143 of 18000 (34.1%)")
    chk("Cancelled+Returned revenue share", f"{100 * cr['revenue'].sum() / sales['revenue'].sum():.1f}%", "17.2%")
    chk("campaign clicks>impressions", int((campaigns_clean["clicks"] > campaigns_clean["impressions"]).sum()), 84)
    chk("campaign conversions>clicks", int((campaigns_clean["conversions"] > campaigns_clean["clicks"]).sum()), 88)
    chk("true_stockout", int(inventory_clean["true_stockout"].sum()), 10)
    chk("at_risk", int(inventory_clean["at_risk"].sum()), 717)
    chk("region_winners cells", int(len(region_winners)), "about 863")
    share = region_winners["winner"].value_counts(normalize=True).mul(100).round(1).to_dict()
    chk("winner share", share, "North 23%, others 18-20%")

    print("\n" + "=" * 78)
    print("BUILD_DB CHECKS: actual vs expected")
    print("=" * 78)
    print(f"{'CHECK':<38}{'ACTUAL':<24}{'EXPECTED':<16}")
    print("-" * 78)
    for label, actual, expected in rows:
        print(f"{label:<38}{str(actual):<24}{str(expected):<16}")
    print("=" * 78)

    print("\nRegion profit per completed sale:")
    print(comp.groupby("region")["profit"].mean().round(2).to_string())
    return rows


def main():
    config.ensure_dirs()
    raw = read_raw()

    sales = build_sales_clean(raw["transactions"], raw["products"], raw["customers"])
    inventory_clean = build_inventory_clean(raw["inventory"])
    print(f"DATA_DIR (read-only) = {config.DATA_DIR}")
    config.assert_raw_rows()

    campaigns_clean = build_campaigns_clean(raw["campaigns"])
    products = raw["products"].copy()
    products["neg_margin_product"] = (products["unit_cost"] >= products["list_price"]).astype(int)

    region_stats = build_region_stats(sales)
    region_winners = build_region_winners(sales)

    con = config.get_connection()
    try:
        raw["customers"].to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        raw["transactions"].to_sql("transactions", con, if_exists="replace", index=False)
        raw["inventory"].to_sql("inventory", con, if_exists="replace", index=False)
        raw["campaigns"].to_sql("campaigns", con, if_exists="replace", index=False)

        sales.to_sql("sales_clean", con, if_exists="replace", index=False)
        inventory_clean.to_sql("inventory_clean", con, if_exists="replace", index=False)
        campaigns_clean.to_sql("campaigns_clean", con, if_exists="replace", index=False)
        region_stats.to_sql("region_stats", con, if_exists="replace", index=False)
        region_winners.to_sql("region_winners", con, if_exists="replace", index=False)

        cur = con.cursor()
        for table in ("sales_clean", "inventory_clean"):
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_product_id ON {table}(product_id)")
        con.commit()
    finally:
        con.close()

    report(sales, products, inventory_clean, campaigns_clean, region_winners)
    print(f"\nWrote {config.DB_PATH}")
    print("Tables:", "customers, products, transactions, inventory, campaigns, "
                     "sales_clean, inventory_clean, campaigns_clean, region_stats, region_winners")


if __name__ == "__main__":
    main()
