"""STEP 4: FastAPI app. Bundle loads once at startup; TreeExplainer is built at startup, never pickled."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import config, rag

STATE = {"bundle": None, "explainer": None, "error": None}


@asynccontextmanager
async def lifespan(app):
    try:
        STATE["bundle"] = rag.load_bundle()
        import shap

        STATE["explainer"] = shap.TreeExplainer(STATE["bundle"]["xgb_reg"])
        print("Bundle + TreeExplainer loaded.")
    except Exception as exc:  # keep the app up so /health can explain the fix
        STATE["error"] = str(exc)
        print(f"STARTUP WARNING: {exc}")
    yield
    STATE["bundle"] = None
    STATE["explainer"] = None


app = FastAPI(title="Money Leak", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def require_bundle():
    if STATE["bundle"] is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model bundle not loaded.",
                "detail": STATE["error"] or "missing bundle",
                "fix": "Run train first: python -m ml_model.train",
            },
        )
    return None


# ------------------------------------------------------------------ Pydantic
class PredictRequest(BaseModel):
    product_id: str = Field(..., description="e.g. PR0001")
    discount_pct: float = Field(0.0, ge=0.0, le=1.0)
    quantity: int = Field(1, ge=1)
    channel: str = "Web"
    customer_segment: str = "Loyal"
    month: int = Field(11, ge=1, le=12)


class AskRequest(BaseModel):
    question: str


class RegionPlanRequest(BaseModel):
    product_id: str = Field(..., description="e.g. PR0001")
    discount_pct: float = Field(0.0, ge=0.0, le=1.0)
    quantity: int = Field(1, ge=1)
    channel: str = "Web"
    customer_segment: str = "Loyal"
    month: int = Field(11, ge=1, le=12)
    region: str | None = Field(None, description="None = rank all five regions")


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "bundle_loaded": STATE["bundle"] is not None,
        "explainer_ready": STATE["explainer"] is not None,
        "startup_error": STATE["error"],
        "db": str(config.DB_PATH),
    }


@app.get("/api/options")
def options():
    blocked = require_bundle()
    if blocked:
        return blocked
    products = rag.query(
        "SELECT product_id, category, list_price, unit_cost FROM products ORDER BY product_id"
    ).to_dict("records")
    categories = rag.query("SELECT DISTINCT category FROM products ORDER BY category")["category"].tolist()
    channels = rag.query("SELECT DISTINCT channel FROM sales_clean ORDER BY channel")["channel"].tolist()
    segments = rag.query("SELECT DISTINCT customer_segment FROM customers ORDER BY customer_segment")["customer_segment"].tolist()
    return {
        "products": products,
        "categories": categories,
        "channels": channels,
        "customer_segments": segments,
        "months": list(range(1, 13)),
        "regions": config.REGIONS,  # display only: region is an output, never an input
        "regions_are_output_only": True,
    }


@app.get("/api/summary")
def summary():
    return rag.leak_summary()


@app.get("/api/products")
def products(category: str | None = None, limit: int = 25):
    sql = (
        "SELECT p.product_id, p.category, p.unit_cost, p.list_price, "
        "COALESCE(SUM(CASE WHEN s.status = 'Completed' THEN s.profit END), 0) AS completed_profit "
        "FROM products p LEFT JOIN sales_clean s ON s.product_id = p.product_id"
    )
    params = []
    if category:
        sql += " WHERE p.category = ?"
        params.append(category)
    sql += " GROUP BY p.product_id ORDER BY completed_profit ASC LIMIT ?"
    params.append(int(limit))
    return {"rows": rag.query(sql, tuple(params)).to_dict("records")}


@app.get("/api/product/{product_id}")
def product(product_id: str, channel: str | None = None, customer_segment: str | None = None):
    try:
        info = rag.product_row(product_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    sales = rag.query("SELECT * FROM sales_clean WHERE product_id = ?", (product_id,))
    inv = rag.query(
        "SELECT * FROM inventory_clean WHERE product_id = ? ORDER BY date DESC LIMIT 1", (product_id,)
    )
    return {
        "product": info,
        "history": rag.get_history(product_id, info["category"], channel, customer_segment),
        "latest_inventory": inv.to_dict("records")[0] if not inv.empty else None,
        "sales_rows": int(len(sales)),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    blocked = require_bundle()
    if blocked:
        return blocked
    try:
        pred = rag.predict_profit(
            req.product_id, req.discount_pct, req.quantity, req.channel, req.customer_segment
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    history = rag.get_history(req.product_id, pred["category"], req.channel, req.customer_segment)
    shap_info = rag.shap_explain(pred, STATE["explainer"])
    region_info = rag.predict_region(
        pred["category"], req.channel, req.customer_segment, req.month
    )

    inv = rag.query(
        "SELECT stock_on_hand, reorder_threshold, at_risk, true_stockout, supplier_delay_days, date "
        "FROM inventory_clean WHERE product_id = ? ORDER BY date DESC LIMIT 1",
        (req.product_id,),
    )
    inventory = inv.to_dict("records")[0] if not inv.empty else None
    history["latest_inventory"] = inventory

    comparison = rag.compare(pred, history)
    explanation = rag.explain_text(pred, history, comparison, shap_info, region_info)

    return {
        "product_id": req.product_id,
        "category": pred["category"],
        "profit": {
            "xgb": pred["xgb_profit"],
            "linear": pred["linear_profit"],
            "tree": pred["tree_profit"],
            "per_unit_xgb": comparison["predicted_profit_per_unit"],
        },
        "loss_flag": pred["loss_flag"],
        "break_even_discount": pred["break_even_discount"],
        "region": region_info,
        "history": history,
        "similar_sales": rag.similar_sales(req.product_id, req.discount_pct, req.quantity),
        "comparison": comparison,
        "shap": shap_info,
        "reasoning": explanation["reasoning"],
        "action": explanation["action"],
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    return rag.ask(req.question)


@app.post("/api/region-plan")
def region_plan(req: RegionPlanRequest):
    blocked = require_bundle()
    if blocked:
        return blocked
    try:
        return rag.region_plan(
            req.product_id, req.discount_pct, req.quantity,
            req.channel, req.customer_segment, req.month, req.region,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/category-cards")
def category_cards():
    return {"cards": rag.category_cards()}


@app.get("/api/campaign-cards")
def campaign_cards():
    return rag.campaign_cards()


@app.get("/api/metrics")
def metrics():
    try:
        return rag.load_metrics()
    except FileNotFoundError:
        return JSONResponse(status_code=503, content={"error": "metrics.json missing", "fix": "run python -m ml_model.train"})


config.ensure_dirs()
if config.CHARTS_DIR.exists():
    app.mount("/charts", StaticFiles(directory=config.CHARTS_DIR), name="charts")
# The built React app in frontend/dist is what gets served at "/".
if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
else:
    print(
        f"FRONTEND NOT BUILT: {config.FRONTEND_DIST} does not exist. "
        "Run: cd frontend && npm install && npm run build"
    )
