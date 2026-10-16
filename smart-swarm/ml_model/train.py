"""STEP 2: train the profit regressors and the (near-chance) winning-region classifiers.

Time split, never random: train = months 1-9, test = months 10-12.
Region is an OUTPUT (classification target), never an input feature.
"""
import json
import sqlite3
import warnings

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from xgboost import XGBClassifier, XGBRegressor

from backend import config

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
RS = config.RANDOM_STATE

PROFIT_FEATURES = config.NUM_COLS + config.CAT_COLS  # 9 features, no region


# ----------------------------------------------------------------- data prep
def load_sales():
    con = config.get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM sales_clean", con)
    finally:
        con.close()
    return df


def build_profit_datasets(df):
    """Dataset 1 = one-hot + scaled (linear family). Dataset 2 = category codes (trees)."""
    dummies = pd.get_dummies(df[PROFIT_FEATURES], columns=config.CAT_COLS, drop_first=True)
    d1_cols = list(dummies.columns)

    cat_maps = {c: sorted(df[c].astype(str).unique()) for c in config.CAT_COLS}
    codes = df[config.NUM_COLS].copy()
    for c in config.CAT_COLS:
        codes[c] = pd.Categorical(df[c].astype(str), categories=cat_maps[c]).codes
    d2_cols = list(codes.columns)

    mask_tr = df["month"].isin(config.TRAIN_MONTHS)
    mask_te = df["month"].isin(config.TEST_MONTHS)

    X1_tr, X1_te = dummies[mask_tr].reset_index(drop=True), dummies[mask_te].reset_index(drop=True)
    X2_tr, X2_te = codes[mask_tr].reset_index(drop=True), codes[mask_te].reset_index(drop=True)
    y_tr = df.loc[mask_tr, "profit"].reset_index(drop=True)
    y_te = df.loc[mask_te, "profit"].reset_index(drop=True)

    scaler = StandardScaler().fit(X1_tr)
    X1_tr_s = pd.DataFrame(scaler.transform(X1_tr), columns=d1_cols)
    X1_te_s = pd.DataFrame(scaler.transform(X1_te), columns=d1_cols)

    return dict(
        d1_cols=d1_cols,
        d2_cols=d2_cols,
        cat_maps=cat_maps,
        scaler=scaler,
        X1_tr=X1_tr_s, X1_te=X1_te_s,
        X2_tr=X2_tr, X2_te=X2_te,
        y_tr=y_tr, y_te=y_te,
    )


def build_winner_datasets(winners):
    """Winning-region data: features category, channel, customer_segment, month."""
    w = winners.copy()
    feat_cols = ["category", "channel", "customer_segment", "month"]
    y = w["winner"].astype(str)

    dummies = pd.get_dummies(w[feat_cols], columns=["category", "channel", "customer_segment"], drop_first=True)
    d1_cols = list(dummies.columns)

    cat_maps = {c: sorted(w[c].astype(str).unique()) for c in ["category", "channel", "customer_segment"]}
    codes = w[["month"]].copy()
    for c in ["category", "channel", "customer_segment"]:
        codes[c] = pd.Categorical(w[c].astype(str), categories=cat_maps[c]).codes
    d2_cols = list(codes.columns)  # must match the matrix the models are fitted on

    mask_tr = w["month"].isin(config.TRAIN_MONTHS)
    mask_te = w["month"].isin(config.TEST_MONTHS)

    classes = sorted(y.unique())
    # XGBClassifier needs integer labels; the index order matches `classes` exactly.
    enc = pd.Categorical(y, categories=classes).codes

    scaler = StandardScaler().fit(dummies[mask_tr])
    X1_tr = pd.DataFrame(scaler.transform(dummies[mask_tr]), columns=d1_cols)
    X1_te = pd.DataFrame(scaler.transform(dummies[mask_te]), columns=d1_cols)

    return dict(
        d1_cols=d1_cols, d2_cols=d2_cols, cat_maps=cat_maps, scaler=scaler,
        X1_tr=X1_tr, X1_te=X1_te,
        X2_tr=codes[mask_tr].reset_index(drop=True), X2_te=codes[mask_te].reset_index(drop=True),
        y_tr=y[mask_tr].reset_index(drop=True), y_te=y[mask_te].reset_index(drop=True),
        y_tr_enc=enc[mask_tr.values], y_te_enc=enc[mask_te.values],
        classes=classes,
    )


# ------------------------------------------------------------- metrics
def reg_scores(model, Xtr, ytr, Xte, yte):
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    baseline = np.full(len(yte), float(np.mean(ytr)))
    return {
        "r2_train": float(r2_score(ytr, ptr)),
        "r2_test": float(r2_score(yte, pte)),
        "mae_test": float(mean_absolute_error(yte, pte)),
        "rmse_test": float(np.sqrt(mean_squared_error(yte, pte))),
        "baseline_mae": float(mean_absolute_error(yte, baseline)),
    }, ptr, pte


def clf_scores(model, Xtr, ytr, Xte, yte, classes):
    pte = model.predict(Xte)
    if np.asarray(pte).dtype.kind in "iu":
        pte = np.array(classes)[np.asarray(pte).astype(int)]
    proba = model.predict_proba(Xte)
    majority = pd.Series(ytr).value_counts().idxmax()
    return {
        "accuracy": float(accuracy_score(yte, pte)),
        "macro_f1": float(f1_score(yte, pte, average="macro")),
        "log_loss": float(log_loss(yte, proba, labels=classes)),
        "uniform_log_loss": float(np.log(len(classes))),
        "majority_baseline": float(accuracy_score(yte, np.full(len(yte), majority))),
        "chance": float(1.0 / len(classes)),
        "roc_auc_ovr": float(roc_auc_score(yte, proba, multi_class="ovr", labels=classes)),
    }


def print_table(title, rows, cols):
    print("\n" + "-" * 92)
    print(title)
    print("-" * 92)
    header = f"{'model':<26}" + "".join(f"{c:<16}" for c in cols)
    print(header)
    for name, m in rows.items():
        line = f"{name:<26}"
        for c in cols:
            v = m[c]
            line += f"{v:<16.4f}" if isinstance(v, float) else f"{str(v):<16}"
        print(line)


# ------------------------------------------------------------------ charts
def save(fig, name):
    path = config.CHARTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_shap(sample_df, values, feature_names, name, plot_type="dot", title=""):
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(values, sample_df, feature_names=feature_names, plot_type=plot_type, show=False)
    if title:
        plt.title(title)
    return save(fig, name)


# ------------------------------------------------------------------- main
def main():
    config.ensure_dirs()
    global shap
    import shap

    sales = load_sales()
    ds = build_profit_datasets(sales)
    metrics = {"dataset": {"train_rows": int(len(ds["y_tr"])), "test_rows": int(len(ds["y_te"]))}}

    # ---------------------------------------------------- A) profit regression
    lin = LinearRegression().fit(ds["X1_tr"], ds["y_tr"])
    tree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=RS).fit(ds["X2_tr"], ds["y_tr"])
    xgb_reg = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.1, eval_metric="rmse",
        random_state=RS, n_jobs=4,
    )
    xgb_reg.fit(
        ds["X2_tr"], ds["y_tr"],
        eval_set=[(ds["X2_tr"], ds["y_tr"]), (ds["X2_te"], ds["y_te"])],
        verbose=False,
    )

    regressors = {"LinearRegression": (lin, "X1"), "DecisionTreeRegressor": (tree, "X2"), "XGBRegressor": (xgb_reg, "X2")}
    reg_metrics, preds = {}, {}
    for name, (m, which) in regressors.items():
        sfx = "1" if which == "X1" else "2"
        mtr, ptr, pte = reg_scores(m, ds[f"X{sfx}_tr"], ds["y_tr"], ds[f"X{sfx}_te"], ds["y_te"])
        reg_metrics[name] = mtr
        preds[name] = pte

    cv_r2 = cross_val_score(
        DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=RS),
        ds["X2_tr"], ds["y_tr"], cv=5, scoring="r2",
    )
    reg_metrics["DecisionTreeRegressor"]["cv_r2_mean"] = float(cv_r2.mean())
    reg_metrics["DecisionTreeRegressor"]["cv_r2_std"] = float(cv_r2.std())
    metrics["profit_regression"] = reg_metrics
    print_table("A) PROFIT REGRESSION (train months 1-9 / test 10-12)",
                reg_metrics, ["r2_train", "r2_test", "mae_test", "rmse_test", "baseline_mae"])

    ev = xgb_reg.evals_result()
    metrics["xgb_reg_evals"] = ev

    # ---------------------------------------------- B) winning-region classifier
    con = config.get_connection()
    try:
        winners = pd.read_sql_query("SELECT * FROM region_winners", con)
    finally:
        con.close()

    wds = build_winner_datasets(winners)
    classes = wds["classes"]

    log_clf = LogisticRegression(max_iter=1000, random_state=RS).fit(wds["X1_tr"], wds["y_tr"])
    tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RS).fit(wds["X2_tr"], wds["y_tr"])
    xgb_clf = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="mlogloss",
        random_state=RS, n_jobs=2,
    )
    xgb_clf.fit(
        wds["X2_tr"], wds["y_tr_enc"],
        eval_set=[(wds["X2_tr"], wds["y_tr_enc"]), (wds["X2_te"], wds["y_te_enc"])],
        verbose=False,
    )

    clf_models = {"LogisticRegression": (log_clf, "X1"), "DecisionTreeClassifier": (tree_clf, "X2"), "XGBClassifier": (xgb_clf, "X2")}
    clf_metrics = {}
    for name, (m, which) in clf_models.items():
        clf_metrics[name] = clf_scores(m, wds[f"{which}_tr"], wds["y_tr"], wds[f"{which}_te"], wds["y_te"], classes)
    best_acc = max(v["accuracy"] for v in clf_metrics.values())
    majority = clf_metrics["XGBClassifier"]["majority_baseline"]
    metrics["winner_region"] = {
        "models": clf_metrics,
        "classes": classes,
        "chance": 1.0 / len(classes),
        "best_test_accuracy": float(best_acc),
        "majority_baseline": float(majority),
        "signal": "none" if best_acc <= majority + 0.02 else "weak",
        "xgb_clf_evals": xgb_clf.evals_result(),
    }
    print_table("B) WINNING REGION (region is an output; expect near-chance accuracy)",
                clf_metrics, ["accuracy", "macro_f1", "log_loss", "uniform_log_loss", "majority_baseline"])
    print(f"\n  chance = {1 / len(classes):.3f}   best test accuracy = {best_acc:.3f}   "
          f"majority baseline = {majority:.3f}   signal = {metrics['winner_region']['signal']}")

    # ----------------------------------------------------------------- C) Kruskal
    comp = sales[sales["status"] == "Completed"]
    def kruskal_p(frame):
        groups = [g["profit"].values for _, g in frame.groupby("region") if len(g) > 1]
        if len(groups) < 2:
            return float("nan")
        return float(kruskal(*groups).pvalue)

    region_tests = {"overall": kruskal_p(comp), "by_category": {}}
    for cat, g in comp.groupby("category"):
        region_tests["by_category"][cat] = kruskal_p(g)
    metrics["region_tests"] = region_tests
    print("\nC) KRUSKAL-WALLIS on completed profit per sale across regions")
    print(f"   overall p = {region_tests['overall']:.4f}")
    for cat, p in region_tests["by_category"].items():
        print(f"   {cat:<14} p = {p:.4f}")

    # ------------------------------------------------------------------ C) SHAP
    sample = ds["X2_tr"].sample(n=min(1500, len(ds["X2_tr"])), random_state=RS)
    explainer = shap.TreeExplainer(xgb_reg)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[0]
    base = float(np.ravel(explainer.expected_value)[0])
    check = base + sv.sum(axis=1)
    err = float(np.max(np.abs(check - xgb_reg.predict(sample))))
    metrics["shap"] = {"base_value": base, "max_base_plus_shap_error": err, "sample_rows": int(len(sample))}

    make_shap(sample, sv, ds["d2_cols"], "shap_summary", "dot", "SHAP beeswarm - XGB profit regressor")
    make_shap(sample, sv, ds["d2_cols"], "shap_bar", "bar", "Mean |SHAP| - XGB profit regressor")
    print(f"\n   SHAP check: base + sum(shap) vs prediction, max abs error = {err:.6f}")

    # winner classifier SHAP, summed across classes
    wsv = shap.TreeExplainer(xgb_clf).shap_values(wds["X2_te"])
    if isinstance(wsv, list):
        wmat = np.abs(np.stack(wsv, axis=0)).sum(axis=0)
    else:
        wmat = np.abs(np.asarray(wsv)).sum(axis=-1)
    fig = plt.figure(figsize=(8, 5))
    order = np.argsort(wmat.mean(axis=0))
    plt.barh([wds["d2_cols"][i] for i in order], wmat.mean(axis=0)[order], color="#7f5af0")
    plt.title("Mean |SHAP| summed over classes - winner classifier (near chance)")
    plt.xlabel("mean |SHAP|")
    save(fig, "shap_winner_bar")

    # ----------------------------------------------------------- D) charts
    fig = plt.figure(figsize=(8, 4.5))
    sns.histplot(sales["profit"], bins=60, color="#2cb67d")
    plt.title("Profit per sale"); plt.xlabel("profit")
    save(fig, "profit_hist")

    num_for_corr = sales[["quantity", "discount_pct", "list_price", "unit_cost", "cost_to_list", "supplier_lead_days", "revenue", "profit"]]
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(num_for_corr.corr(), annot=True, fmt=".2f", cmap="mako")
    plt.title("Correlation heatmap")
    save(fig, "corr_heatmap")

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"LinearRegression": "#7f5af0", "DecisionTreeRegressor": "#2cb67d", "XGBRegressor": "#ff8e3c"}
    for name, pte in preds.items():
        ax.scatter(ds["y_te"], pte, s=4, alpha=0.35, label=name, color=colors[name])
    lim = [float(ds["y_te"].min()), float(ds["y_te"].max())]
    ax.plot(lim, lim, "w--", lw=1)
    ax.set_xlabel("actual profit"); ax.set_ylabel("predicted profit")
    ax.set_title("Predicted vs actual (test months)"); ax.legend()
    save(fig, "pred_vs_actual")

    resid = ds["y_te"] - preds["XGBRegressor"]
    fig = plt.figure(figsize=(8, 4.5))
    sns.histplot(resid, bins=60, color="#ff8e3c")
    plt.title("XGB residuals (test)"); plt.xlabel("actual - predicted")
    save(fig, "residual_hist")

    names = list(reg_metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, [reg_metrics[n]["r2_test"] for n in names], color="#7f5af0")
    axes[0].set_title("Test R2"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, [reg_metrics[n]["mae_test"] for n in names], color="#2cb67d")
    axes[1].axhline(reg_metrics[names[0]]["baseline_mae"], color="w", ls="--", label="baseline MAE")
    axes[1].set_title("Test MAE"); axes[1].tick_params(axis="x", rotation=20); axes[1].legend()
    save(fig, "model_comparison")

    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(ev["validation_0"]["rmse"], label="train")
    plt.plot(ev["validation_1"]["rmse"], label="test")
    plt.xlabel("round"); plt.ylabel("rmse"); plt.title("XGB regressor training curve"); plt.legend()
    save(fig, "xgb_reg_training_curve")

    evc = metrics["winner_region"]["xgb_clf_evals"]
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(evc["validation_0"]["mlogloss"], label="train")
    plt.plot(evc["validation_1"]["mlogloss"], label="test")
    plt.xlabel("round"); plt.ylabel("mlogloss"); plt.title("XGB winner classifier training curve"); plt.legend()
    save(fig, "xgb_clf_training_curve")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [reg_metrics[n]["r2_train"] for n in names], w, label="train R2", color="#2cb67d")
    ax.bar(x + w / 2, [reg_metrics[n]["r2_test"] for n in names], w, label="test R2", color="#ff8e3c")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.legend(); ax.set_title("Overfit check")
    save(fig, "overfit_check")

    imp = xgb_reg.feature_importances_
    order = np.argsort(imp)
    fig = plt.figure(figsize=(8, 5))
    plt.barh([ds["d2_cols"][i] for i in order], imp[order], color="#ff8e3c")
    plt.title("XGB feature importance"); plt.xlabel("importance")
    save(fig, "xgb_feature_importance")

    coefs = lin.coef_
    order = np.argsort(np.abs(coefs))
    fig = plt.figure(figsize=(9, 6))
    plt.barh([ds["d1_cols"][i] for i in order], coefs[order], color="#7f5af0")
    plt.title("Linear coefficients (scaled)"); plt.xlabel("coefficient")
    save(fig, "linear_coefficients")

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(tree, max_depth=3, feature_names=ds["d2_cols"], filled=True, fontsize=8, ax=ax)
    plt.title("DecisionTreeRegressor (first 3 levels)")
    save(fig, "tree_plot")

    fig = plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=comp, x="region", y="profit", color="#7f5af0")
    plt.title(f"Completed profit per sale by region (Kruskal p = {region_tests['overall']:.2f})")
    save(fig, "region_profit_box")

    share = winners["winner"].value_counts(normalize=True).mul(100)
    fig = plt.figure(figsize=(8, 4.5))
    plt.bar(share.index, share.values, color="#2cb67d")
    plt.axhline(20, color="w", ls="--", label="20% (equal chance)")
    plt.ylabel("% of cells won"); plt.title("Region winner share"); plt.legend()
    save(fig, "region_winner_share")

    fig = plt.figure(figsize=(9, 4.5))
    labels = list(clf_metrics.keys()) + ["majority", "chance"]
    vals = [clf_metrics[n]["accuracy"] for n in clf_metrics] + [majority, 1 / len(classes)]
    plt.bar(labels, vals, color=["#7f5af0", "#2cb67d", "#ff8e3c", "#72757e", "#72757e"])
    plt.axhline(1 / len(classes), color="w", ls="--")
    plt.ylabel("test accuracy"); plt.title("Winning-region accuracy vs majority vs chance")
    plt.xticks(rotation=20)
    save(fig, "winner_vs_chance")

    cm_pred = np.array(classes)[xgb_clf.predict(wds["X2_te"]).astype(int)]
    cm = confusion_matrix(wds["y_te"], cm_pred, labels=classes)
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=classes, yticklabels=classes)
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Winner confusion matrix (XGB)")
    save(fig, "winner_confusion_matrix")

    # --------------------------------------------------------------- E) bundle
    bundle = {
        "xgb_reg": xgb_reg, "lin_reg": lin, "tree_reg": tree,
        "xgb_clf": xgb_clf, "log_clf": log_clf, "tree_clf": tree_clf,
        "scaler": ds["scaler"],
        "d1_cols": ds["d1_cols"], "d2_cols": ds["d2_cols"], "cat_maps": ds["cat_maps"],
        "num_cols": config.NUM_COLS, "cat_cols": config.CAT_COLS,
        "regions": config.REGIONS,
        "winner_d1_cols": wds["d1_cols"], "winner_d2_cols": wds["d2_cols"],
        "winner_cat_maps": wds["cat_maps"], "winner_classes": classes,
    }
    # ------------------------------------------------------- F) model registry
    reg = reg_metrics
    wm = metrics["winner_region"]["models"]
    registry = [
        {"model": "LinearRegression", "task": "profit", "dataset": "Dataset 1",
         "features": "9 features", "target": "profit", "metric": "R2, MAE",
         "role": "baseline, shows non-linearity",
         "r2_test": reg["LinearRegression"]["r2_test"], "mae_test": reg["LinearRegression"]["mae_test"]},
        {"model": "DecisionTreeRegressor", "task": "profit", "dataset": "Dataset 2",
         "features": "9 features", "target": "profit", "metric": "R2, MAE",
         "role": "interpretable second opinion",
         "r2_test": reg["DecisionTreeRegressor"]["r2_test"], "mae_test": reg["DecisionTreeRegressor"]["mae_test"]},
        {"model": "XGBRegressor", "task": "profit", "dataset": "Dataset 2",
         "features": "9 features", "target": "profit", "metric": "R2, MAE",
         "role": "PRODUCTION profit model, explained by SHAP",
         "r2_test": reg["XGBRegressor"]["r2_test"], "mae_test": reg["XGBRegressor"]["mae_test"]},
        {"model": "LogisticRegression", "task": "winning region", "dataset": "region_winners",
         "features": "4 features", "target": "winner", "metric": "accuracy, log-loss",
         "role": "baseline",
         "accuracy": wm["LogisticRegression"]["accuracy"], "log_loss": wm["LogisticRegression"]["log_loss"]},
        {"model": "DecisionTreeClassifier", "task": "winning region", "dataset": "region_winners",
         "features": "4 features", "target": "winner", "metric": "accuracy, log-loss",
         "role": "baseline",
         "accuracy": wm["DecisionTreeClassifier"]["accuracy"], "log_loss": wm["DecisionTreeClassifier"]["log_loss"]},
        {"model": "XGBClassifier", "task": "winning region", "dataset": "region_winners",
         "features": "4 features", "target": "winner", "metric": "accuracy, log-loss",
         "role": "PRODUCTION region model (honest verdict rule)",
         "accuracy": wm["XGBClassifier"]["accuracy"], "log_loss": wm["XGBClassifier"]["log_loss"]},
    ]
    metrics["model_registry"] = registry
    metrics["not_trained"] = "returns, cancellations, stockouts (no signal)"

    print("\nF) MODEL REGISTRY")
    print(f"  {'model':<24}{'task':<16}{'dataset':<16}{'features':<10}{'target':<9}{'metric':<18}role")
    for r in registry:
        print(f"  {r['model']:<24}{r['task']:<16}{r['dataset']:<16}{r['features']:<10}"
              f"{r['target']:<9}{r['metric']:<18}{r['role']}")
    print(f"  not trained: {metrics['not_trained']}")

    # ------------------------------------------------------------ G) self-check
    con = sqlite3.connect(config.DB_PATH)
    winners_rows = int(pd.read_sql_query("SELECT COUNT(*) AS n FROM region_winners", con).iloc[0, 0])
    con.close()

    mask_tr = sales["month"].isin(config.TRAIN_MONTHS)
    split_ok = (
        int(mask_tr.sum()) == len(ds["y_tr"])
        and set(sales.loc[mask_tr, "month"].unique()) <= set(config.TRAIN_MONTHS)
        and set(sales.loc[~mask_tr, "month"].unique()) <= set(config.TEST_MONTHS)
    )

    forbidden = {"region", "unit_price", "revenue", "status", "product_id", "customer_id", "transaction_id"}
    profit_feats = set(ds["d2_cols"]) | set(ds["d1_cols"])
    winner_feats = set(wds["d2_cols"]) | set(wds["d1_cols"])
    winner_base = {"category", "channel", "customer_segment", "month"}
    # a winner row carries no sale-level column, so none of these may leak in as a feature
    winner_leak = {
        "profit", "revenue", "quantity", "discount_pct", "unit_cost", "list_price",
        "cost_to_list", "supplier_lead_days", "region", "status", "unit_price",
    }

    checks = [
        ("profit features contain no region, unit_price, revenue, status or IDs",
         not (profit_feats & forbidden)),
        ("the winner classifier was trained only from region_winners",
         len(wds["y_tr"]) + len(wds["y_te"]) == winners_rows),
        ("winner features are the 4 declared columns and carry no sale-level leakage",
         winner_base <= winner_feats and not (winner_feats & winner_leak)),
        ("time split is months 1-9 train and 10-12 test",
         split_ok),
        ("both XGB models used eval_set",
         bool(xgb_reg.evals_result()) and bool(xgb_clf.evals_result())),
        ("the bundle holds all six models",
         all(k in bundle for k in ("lin_reg", "tree_reg", "xgb_reg", "log_clf", "tree_clf", "xgb_clf"))),
        ("no return, cancel or stockout model exists",
         not any(k in bundle for k in ("return_model", "cancel_model", "stockout_model"))),
    ]
    print("\nG) SELF-CHECK")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    metrics["self_checks"] = [{"check": label, "result": "PASS" if ok else "FAIL"} for label, ok in checks]
    metrics["self_checks_all_pass"] = all(ok for _, ok in checks)
    if not metrics["self_checks_all_pass"]:
        raise SystemExit("SELF-CHECK FAILED - fix train.py and re-run. Do not start the API.")

    joblib.dump(bundle, config.BUNDLE_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\nSaved bundle -> {config.BUNDLE_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")
    print(f"Saved {len(list(config.CHARTS_DIR.glob('*.png')))} charts -> {config.CHARTS_DIR}")


if __name__ == "__main__":
    main()
