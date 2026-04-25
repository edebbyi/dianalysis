"""
Dianalysis Streamlit app (interactive scoring + recommendation UI).

Why:
- Keep all user-facing rendering and input flows in one place.
- Keep model/recommendation business logic out of the UI layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st
from dianalysis.run_config import cfg_get, load_runtime_config


def _parse_app_config_args() -> argparse.Namespace:
    """Parse optional app-level config/profile arguments after `streamlit run app.py -- ...`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=Path(os.getenv("DIANALYSIS_CONFIG", "configs/base.toml")))
    parser.add_argument(
        "--profile",
        type=Path,
        default=(Path(os.environ["DIANALYSIS_PROFILE"]) if os.getenv("DIANALYSIS_PROFILE") else None),
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


_APP_CFG_ARGS = _parse_app_config_args()
_APP_CFG = load_runtime_config(_APP_CFG_ARGS.config, _APP_CFG_ARGS.profile)

# Export active config/profile so downstream modules can read the same runtime profile.
os.environ["DIANALYSIS_CONFIG"] = str(_APP_CFG_ARGS.config)
if _APP_CFG_ARGS.profile:
    os.environ["DIANALYSIS_PROFILE"] = str(_APP_CFG_ARGS.profile)
else:
    os.environ.pop("DIANALYSIS_PROFILE", None)

# Apply retrieval defaults from config unless caller already set explicit env vars.
if "DIANALYSIS_RETRIEVAL_BACKEND" not in os.environ:
    os.environ["DIANALYSIS_RETRIEVAL_BACKEND"] = str(cfg_get(_APP_CFG, "retrieval", "backend", default="qdrant"))
if "QDRANT_URL" not in os.environ:
    os.environ["QDRANT_URL"] = str(cfg_get(_APP_CFG, "retrieval", "qdrant_url", default="http://localhost:6333"))
if "DIANALYSIS_QDRANT_COLLECTION" not in os.environ:
    os.environ["DIANALYSIS_QDRANT_COLLECTION"] = str(
        cfg_get(_APP_CFG, "retrieval", "qdrant_collection", default="dianalysis_products")
    )
if "DIANALYSIS_EMBED_MODEL" not in os.environ:
    os.environ["DIANALYSIS_EMBED_MODEL"] = str(
        cfg_get(_APP_CFG, "retrieval", "embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    )

_MODEL_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from dianalysis.model import compute_model_fingerprint, load_model, generate_synthetic_data, train_model
    from dianalysis.scoring import score_item, score_by_barcode
except ModuleNotFoundError as e:
    _MODEL_IMPORT_ERROR = e

ARTIFACTS_DIR = str(cfg_get(_APP_CFG, "paths", "artifacts_dir", default="artifacts"))
CLEAN_CSV_PATH = str(cfg_get(_APP_CFG, "paths", "input_csv", default="data/products_off_clean.csv"))
SCORED_CSV_PATH = str(cfg_get(_APP_CFG, "paths", "scored_csv", default="data/products_off_clean_scored.csv"))
MANUAL_DEFAULTS: dict[str, Any] = {
    "name": "Frosted Cereal",
    "brand": "",
    "category": "cereal",
    "serving_g": 40.0,
    "calories": 160.0,
    "fat_g": 2.0,
    "carbs_g": 37.0,
    "fiber_g": 3.0,
    "sugar_g": 14.0,
    "added_sugar_g": 10.0,
    "protein_g": 3.0,
    "sodium_mg": 240.0,
}
MANUAL_CATEGORY_OPTIONS = ["cereal", "bread", "snack", "drink", "dairy", "grain"]

try:
    from streamlit.runtime.scriptrunner_utils.exceptions import RerunException
    from streamlit.runtime.scriptrunner_utils.script_requests import RerunData

    def _request_rerun() -> None:
        raise RerunException(RerunData())

except ImportError:  # fallback when running in environments without rerun helpers
    def _request_rerun() -> None:
        return

# Page configuration
st.set_page_config(
    page_title="Dianalysis - Food Risk Scoring",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    section.main > div.block-container {
        padding-top: 1.1rem;
    }
    .risk-low { color: #22c55e; font-weight: 700; }
    .risk-medium { color: #f59e0b; font-weight: 700; }
    .risk-high { color: #ef4444; font-weight: 700; }
    .result-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0 16px 0;
        background: rgba(15, 23, 42, 0.02);
    }
    .score-big {
        font-size: 2.4rem;
        line-height: 1;
        margin: 8px 0;
        font-weight: 800;
    }
    .score-subtle {
        color: rgba(148, 163, 184, 0.95);
        font-size: 0.92rem;
        margin-top: 6px;
    }
    .driver-chip {
        display: inline-block;
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 999px;
        padding: 3px 10px;
        margin: 4px 6px 0 0;
        font-size: 0.80rem;
        background-color: transparent;
    }
    .alternative-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        padding: 12px;
        margin: 10px 0;
        background: rgba(15, 23, 42, 0.015);
    }
    .alternative-card h4 {
        margin: 0 0 2px 0;
        font-size: 0.98rem;
    }
    .alt-meta {
        color: rgba(148, 163, 184, 0.95);
        font-size: 0.86rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

if _MODEL_IMPORT_ERROR is not None:
    missing_pkg = getattr(_MODEL_IMPORT_ERROR, "name", "a required package")
    st.error(
        f"Missing Python package: `{missing_pkg}`. "
        "This app needs the full project dependencies in the same Python environment."
    )
    st.info(f"Python executable in use: `{sys.executable}`")
    st.code(f"{sys.executable} -m pip install -r requirements.txt", language="bash")
    st.caption("If you prefer Docker, run: `make app`.")
    st.stop()


@st.cache_resource
def load_trained_model(_artifact_state: tuple[Any, ...]) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Load or train the model."""
    artifacts_dir = ARTIFACTS_DIR

    has_logreg_artifacts = os.path.exists(os.path.join(artifacts_dir, "model.joblib"))
    has_xgb_artifacts = (
        os.path.exists(os.path.join(artifacts_dir, "meta.joblib"))
        and os.path.exists(os.path.join(artifacts_dir, "preprocessor.joblib"))
        and os.path.exists(os.path.join(artifacts_dir, "xgb_model.json"))
    )

    if has_logreg_artifacts or has_xgb_artifacts:
        try:
            model, meta = load_model(artifacts_dir)
            saved_metrics = meta.get("metrics") if isinstance(meta, dict) else None
            return model, meta, saved_metrics
        except Exception as e:
            st.warning(f"Could not load existing model: {e}. Training new model...")
    
    # Train new model
    with st.spinner("Training model on synthetic data..."):
        df = generate_synthetic_data(n=1000, random_state=42)
        model, metrics = train_model(df, artifacts_dir=artifacts_dir)
        meta = {"num_cols": None, "cat_cols": None, "categories": None}
    return model, meta, metrics


def _artifact_state(artifacts_dir: str = ARTIFACTS_DIR) -> tuple[Any, ...]:
    """
    Build a cache key from artifact file state so Streamlit refreshes model/metrics
    when artifacts are retrained or replaced.
    """
    tracked = [
        "meta.joblib",
        "model.joblib",
        "preprocessor.joblib",
        "xgb_model.json",
    ]
    artifact_state: list[tuple[str, int, int] | tuple[str, None, None]] = []
    for name in tracked:
        path = os.path.join(artifacts_dir, name)
        if os.path.exists(path):
            artifact_state.append((name, int(os.path.getmtime(path)), os.path.getsize(path)))
        else:
            artifact_state.append((name, None, None))
    return tuple(artifact_state)


def _infer_model_type(meta: dict[str, Any] | None, artifacts_dir: str = ARTIFACTS_DIR) -> str:
    """Resolve active model type from metadata first, then artifact files."""
    if isinstance(meta, dict):
        raw = str(meta.get("model_type", "")).strip().lower()
        if raw in {"logreg", "xgboost"}:
            return raw

    has_xgb = (
        os.path.exists(os.path.join(artifacts_dir, "meta.joblib"))
        and os.path.exists(os.path.join(artifacts_dir, "preprocessor.joblib"))
        and os.path.exists(os.path.join(artifacts_dir, "xgb_model.json"))
    )
    if has_xgb:
        return "xgboost"
    if os.path.exists(os.path.join(artifacts_dir, "model.joblib")):
        return "logreg"
    return "unavailable"


def _metrics_timestamp_utc(meta: dict[str, Any] | None, artifacts_dir: str = ARTIFACTS_DIR) -> tuple[str | None, str | None]:
    """Return best-available metrics timestamp in UTC and where it came from."""
    if isinstance(meta, dict):
        recommendation_eval = meta.get("recommendation_eval", {})
        evaluated_at_utc = recommendation_eval.get("evaluated_at_utc") if isinstance(recommendation_eval, dict) else None
        if isinstance(evaluated_at_utc, str) and evaluated_at_utc.strip():
            return evaluated_at_utc, "recommendation_eval.evaluated_at_utc"

    meta_path = os.path.join(artifacts_dir, "meta.joblib")
    if os.path.exists(meta_path):
        mtime_utc = datetime.fromtimestamp(os.path.getmtime(meta_path), tz=timezone.utc).isoformat()
        return mtime_utc, "meta.joblib mtime"
    return None, None


@st.cache_data
def load_candidates_data() -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Load candidate products for alternatives from OpenFoodFacts."""
    clean_csv = CLEAN_CSV_PATH
    scored_csv = SCORED_CSV_PATH
    
    messages = []
    if os.path.exists(scored_csv):
        try:
            df = pd.read_csv(scored_csv, dtype={"upc": str})
            messages.append(("info", "✅ Using pre-scored alternatives database", "off_fetch_scored_csv"))
            return df, messages
        except Exception as e:
            st.warning(f"Could not load pre-scored alternatives database: {e}")

    if os.path.exists(clean_csv):
        try:
            df = pd.read_csv(clean_csv, dtype={"upc": str})
            messages.append(("info", "✅ Using real OpenFoodFacts database for alternatives", "off_fetch_csv"))
            return df, messages
        except Exception as e:
            st.warning(f"Could not load alternatives database: {e}")
    
    # Try to fetch real data from OpenFoodFacts for common categories
    messages.append(("info", "📡 Loading real alternatives from OpenFoodFacts (this may take a moment)...", "off_fetch_info"))
    
    try:
        from dianalysis.off_pipeline import fetch_category_products
        
        # Fetch products from common categories
        categories = ["nuts", "cereals", "breads", "snacks", "dairy", "beverages"]
        all_products = []
        
        for category in categories:
            try:
                products = fetch_category_products(category, limit=50)
                all_products.extend(products)
            except Exception:
                continue
        
        if all_products:
            df = pd.DataFrame(all_products)
            messages.append(("success", f"✅ Loaded {len(df)} real products from OpenFoodFacts", "off_fetch_success"))
            return df, messages
        else:
            st.warning("⚠️ Could not fetch real data, using synthetic alternatives")
    except Exception as e:
        st.warning(f"⚠️ OpenFoodFacts fetch failed: {e}, using synthetic alternatives")
    
    # Fallback: use synthetic data
    df = generate_synthetic_data(n=500, random_state=42)
    messages.append(("info", "📊 Using synthetic data for alternatives (demo mode)", "off_fetch_synthetic"))
    return df, messages


@st.cache_resource
def warmup_retrieval() -> str:
    """
    Warm retrieval dependencies at app startup.

    Why:
    - Streamlit app instances can cold-start after idle/redeploy.
    - A tiny warm-up (embed + one Qdrant call) reduces first-user latency.
    """
    enabled = str(os.getenv("DIANALYSIS_WARMUP_RETRIEVAL", "1")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return "disabled"

    backend = str(os.getenv("DIANALYSIS_RETRIEVAL_BACKEND", "qdrant")).strip().lower()
    if backend != "qdrant":
        return "skipped (backend)"

    try:
        from dianalysis.recommendation.vector_client import (
            collection_name,
            embedder,
            qdrant_client,
            retrieval_enabled,
        )

        if not retrieval_enabled():
            return "skipped (deps)"

        emb = embedder()
        vec = emb.encode(["dianalysis warmup"], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

        client = qdrant_client()
        collection = collection_name()
        # Collection check warms client auth/session and validates target collection.
        client.get_collection(collection_name=collection)
        # Tiny query warms vector retrieval path.
        client.query_points(
            collection_name=collection,
            query=vec,
            limit=1,
            with_payload=False,
        )
        return "ok"
    except Exception:
        return "failed"


def _show_dismissable_message(key: str, text: str, style: str = "info") -> None:
    """Show a message with a dismiss button stored in session state."""
    hidden_flag = f"hide_message_{key}"
    if st.session_state.get(hidden_flag):
        return
    container = st.container()
    if style == "info":
        container.info(text)
    else:
        container.success(text)
    if container.button("Dismiss", key=f"dismiss_{key}"):
        st.session_state[hidden_flag] = True
        container.empty()
        _request_rerun()


def get_risk_class(score: float | int) -> str:
    """Get CSS class for risk score."""
    if score < 30:
        return "risk-low"
    elif score < 70:
        return "risk-medium"
    else:
        return "risk-high"


def get_risk_tier(score: float | int) -> str:
    """Return user-facing risk tier text."""
    if score < 30:
        return "Low"
    if score < 70:
        return "Moderate"
    return "High"


def get_risk_takeaway(score: float | int) -> str:
    """Return one-line scientifically grounded takeaway text."""
    if score < 30:
        return "Lower screening risk for this serving based on carb, sugar, sodium, fiber, and protein signals."
    if score < 70:
        return "Mixed screening signal: some risk nutrients are present, with partial protective offsets."
    return "Higher screening risk for this serving, driven by high-load nutrients and limited protective offsets."


def _display_to_float(display_dict: dict[str, Any], key: str) -> float | None:
    """Parse a display field like '14.0g' or '240mg' into a float."""
    raw = display_dict.get(key)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or "not listed" in text:
        return None
    cleaned = (
        text.replace("mg", "")
        .replace("g", "")
        .replace("kcal", "")
        .replace("<", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def _reset_manual_form() -> None:
    """Reset manual-entry widgets to defaults and clear displayed result."""
    st.session_state["manual_name"] = MANUAL_DEFAULTS["name"]
    st.session_state["manual_brand"] = MANUAL_DEFAULTS["brand"]
    st.session_state["manual_category"] = MANUAL_DEFAULTS["category"]
    st.session_state["manual_serving_g"] = MANUAL_DEFAULTS["serving_g"]
    st.session_state["manual_calories"] = MANUAL_DEFAULTS["calories"]
    st.session_state["manual_fat_g"] = MANUAL_DEFAULTS["fat_g"]
    st.session_state["manual_carbs_g"] = MANUAL_DEFAULTS["carbs_g"]
    st.session_state["manual_fiber_g"] = MANUAL_DEFAULTS["fiber_g"]
    st.session_state["manual_sugar_g"] = MANUAL_DEFAULTS["sugar_g"]
    st.session_state["manual_added_sugar_g"] = MANUAL_DEFAULTS["added_sugar_g"]
    st.session_state["manual_protein_g"] = MANUAL_DEFAULTS["protein_g"]
    st.session_state["manual_sodium_mg"] = MANUAL_DEFAULTS["sodium_mg"]
    st.session_state["last_result"] = None


def display_nutrition_table(display_dict: dict[str, Any]) -> None:
    """Display nutrition facts table."""
    st.subheader("Nutrition Facts")
    
    cols = st.columns(4)
    nutrients = [
        ("Calories", "calories"),
        ("Carbs", "carbs_g"),
        ("Fiber", "fiber_g"),
        ("Sugar", "sugar_g"),
        ("Added Sugar", "added_sugar_g"),
        ("Protein", "protein_g"),
        ("Fat", "fat_g"),
        ("Sodium", "sodium_mg"),
    ]
    
    for i, (label, key) in enumerate(nutrients):
        col = cols[i % 4]
        value = display_dict.get(key, "—")
        col.metric(label, value)


def display_alternatives(
    alternatives: list[dict[str, Any]],
    *,
    current_risk: float | int | None,
    current_net: float | None,
    current_fiber: float | None,
) -> None:
    """Display alternative recommendations as data-rich cards."""
    if not alternatives:
        st.info("No lower-risk alternatives were found in the current dataset for this food group.")
        return

    st.subheader("Better Alternatives")
    st.caption("Same-group swaps ranked to reduce glycemic load while keeping the food context familiar.")

    tier_order = {"best": 0, "better": 1, "good": 2}

    def _as_float(val: Any, default: float) -> float:
        """Safely cast a value to float without turning 0 into the default."""
        try:
            if val is None:
                return default
            return float(val)
        except Exception:
            return default

    def _sort_key(alt: dict[str, Any]) -> tuple[Any, ...]:
        alt_risk = _as_float(alt.get("risk_score"), 100.0)
        alt_net = float(alt.get("net_carbs_g", 0.0) or 0.0)
        alt_fiber = float(alt.get("fiber_g", 0.0) or 0.0)
        tier_rank = tier_order.get(str(alt.get("tier", "good")).lower(), 99)
        return (alt_risk, alt_net, -alt_fiber, tier_rank)

    ranked_alternatives = sorted(alternatives, key=_sort_key)

    choice_labels = ("Best", "Better", "Good")

    for idx, alt in enumerate(ranked_alternatives):
        tier = choice_labels[idx] if idx < len(choice_labels) else "Good"
        alt_name = str(alt.get("name", "Unknown"))
        alt_brand = alt.get("brand", "N/A")
        alt_risk = _as_float(alt.get("risk_score"), 100.0)
        alt_risk_value = "<1" if alt_risk < 1 else f"{int(round(alt_risk))}"
        alt_risk_tier = get_risk_tier(alt_risk)
        alt_risk_class = get_risk_class(alt_risk)
        alt_net = float(alt.get("net_carbs_g", 0.0) or 0.0)
        alt_fiber = float(alt.get("fiber_g", 0.0) or 0.0)

        net_delta_txt = "net carbs: n/a"
        if current_net is not None:
            delta_net = alt_net - float(current_net)
            if delta_net < 0:
                net_delta_txt = f"net carbs: {delta_net:.1f}g"
            else:
                net_delta_txt = f"net carbs: +{delta_net:.1f}g"

        fiber_delta_txt = "fiber: n/a"
        if current_fiber is not None:
            delta_fiber = alt_fiber - float(current_fiber)
            if delta_fiber >= 0:
                fiber_delta_txt = f"fiber: +{delta_fiber:.1f}g"
            else:
                fiber_delta_txt = f"fiber: {delta_fiber:.1f}g"

        st.markdown(
            f"""
            <div class="alternative-card">
                <h4>{tier} Choice: {alt_name}</h4>
                <div class="alt-meta">Brand: {alt_brand}</div>
                <div><strong>Risk score:</strong> <span class="{alt_risk_class}">{alt_risk_value}</span></div>
                <div class="alt-meta">Tier: <span class="{alt_risk_class}">{alt_risk_tier}</span></div>
                <div style="margin-top:4px;">
                    <span class="driver-chip">{net_delta_txt}</span>
                    <span class="driver-chip">{fiber_delta_txt}</span>
                </div>
                <div style="margin-top:8px;"><strong>Why better:</strong> {alt.get('why', 'Lower risk in same category')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result_columns(result: dict[str, Any] | None) -> None:
    """Render item details on left and alternatives on right."""
    if not result:
        return
    
    insufficient_data = bool(result.get("insufficient_data"))
    risk_score_raw = float(result.get("risk_score", 0) or 0)
    risk_score = float(result.get("risk_score_display", risk_score_raw) or risk_score_raw)
    risk_class = get_risk_class(risk_score)
    risk_tier = get_risk_tier(risk_score)
    takeaway = get_risk_takeaway(risk_score)
    data_confidence = str(result.get("data_confidence", "high") or "high").lower()
    confidence_notes = [str(x) for x in (result.get("data_confidence_notes") or []) if str(x).strip()]
    display_dict = result.get("display", {}) or {}
    current_net_val = float(result.get("item_net_carbs_g", 0.0) or 0.0)
    current_net: float | None = current_net_val
    if current_net_val <= 0:
        carbs = _display_to_float(display_dict, "carbs_g")
        fiber = _display_to_float(display_dict, "fiber_g")
        if carbs is not None and fiber is not None:
            current_net = max(carbs - fiber, 0.0)
        else:
            current_net = None
    current_fiber = result.get("item_fiber_g")
    if current_fiber is None:
        current_fiber = _display_to_float(display_dict, "fiber_g")
    top_drivers = result.get("reasons", [])[:3]

    st.divider()
    cols = st.columns([3, 2])
    with cols[0]:
        if insufficient_data:
            reason = result.get("insufficient_data_reason") or "Not enough nutrition information was available."
            st.markdown(
                f"""
                <div class="result-card">
                    <div><strong>Risk Score</strong></div>
                    <div class="score-big">Not available</div>
                    <div class="score-subtle">{reason}</div>
                    <div class="score-subtle">You can still review alternatives below, or use Manual Entry for a full score.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <div><strong>Risk Score</strong></div>
                    <div class="score-big {risk_class}">{result.get('risk_display', int(round(risk_score)))}</div>
                    <div><strong>Tier:</strong> <span class="{risk_class}">{risk_tier}</span></div>
                    <div class="score-subtle">{takeaway}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if data_confidence == "low":
                if confidence_notes:
                    details = "; ".join(confidence_notes[:2])
                    if len(confidence_notes) > 2:
                        details += "; ..."
                    st.warning(f"Data confidence is low: {details}.")
                else:
                    st.warning("Data confidence is low: some critical nutrition fields were missing.")
        
        st.subheader(f"{result.get('item_name', 'Unknown Food')}")
        if result.get('item_brand'):
            st.caption(f"Brand: {result['item_brand']}")
        category_main = str(result.get("item_category_main", result.get("item_category", "Unknown")) or "Unknown")
        alt_group = str(result.get("item_alt_group", "") or "").strip()
        alt_group_fine = str(result.get("item_alt_group_fine", "") or "").strip()

        caption_parts = [f"Category: {category_main}"]
        if alt_group and alt_group.lower() not in {"unknown", "none", "nan"}:
            if alt_group.lower() != category_main.lower():
                caption_parts.append(f"Group: {alt_group}")
        if alt_group_fine and alt_group_fine.lower() not in {"unknown", "none", "nan"}:
            type_display = alt_group_fine.split(":", 1)[1] if ":" in alt_group_fine else alt_group_fine
            if type_display:
                caption_parts.append(f"Type: {type_display}")

        st.caption(" | ".join(caption_parts))

        if not insufficient_data:
            st.subheader("Top Drivers")
            if top_drivers:
                for reason in top_drivers:
                    st.markdown(f"<span class='driver-chip'>{reason}</span>", unsafe_allow_html=True)
            else:
                st.write("No drivers available.")

        display_nutrition_table(display_dict)

        if not insufficient_data:
            with st.expander("Show full explanation"):
                for reason in result.get("reasons", []):
                    st.write(f"• {reason}")

        if result.get("notes"):
            with st.expander("ℹ️ Data Notes"):
                for note in result["notes"]:
                    st.info(note)

    with cols[1]:
        display_alternatives(
            result.get("alternatives", []),
            current_risk=(None if insufficient_data else risk_score_raw),
            current_net=current_net,
            current_fiber=float(current_fiber) if current_fiber is not None else None,
        )


def display_alternatives_placeholder() -> None:
    cols = st.columns([3, 2])
    with cols[0]:
        st.write("")
    with cols[1]:
        st.subheader("Better Alternatives")
        st.info("Submit a food item to score it and see healthier swaps appear here.")


def main() -> None:
    """Main app function."""
    
    # Load model and data
    model, meta, metrics = load_trained_model(_artifact_state())
    if isinstance(meta, dict):
        os.environ["DIANALYSIS_MODEL_TYPE"] = str(meta.get("model_type", "") or "").strip().lower()
    try:
        os.environ["DIANALYSIS_MODEL_FINGERPRINT"] = compute_model_fingerprint(ARTIFACTS_DIR, meta=meta if isinstance(meta, dict) else None)
    except Exception:
        os.environ.pop("DIANALYSIS_MODEL_FINGERPRINT", None)
    df_candidates, candidate_messages = load_candidates_data()
    _ = warmup_retrieval()
    for style, text, key in candidate_messages:
        _show_dismissable_message(key, text, style=style)
    
    # Header (kept intentionally minimal so input is visible faster)
    st.title("🍎 Dianalysis")
    st.markdown("### Diabetes-aware food scoring")
    st.caption("Risk score is 0–100 (lower is better).")

    recommendation_eval = meta.get("recommendation_eval", {}) if isinstance(meta, dict) else {}
    model_type = _infer_model_type(meta, ARTIFACTS_DIR)
    if isinstance(meta, dict):
        meta["model_type"] = model_type
    metrics_timestamp, metrics_timestamp_source = _metrics_timestamp_utc(meta, ARTIFACTS_DIR)
    
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    # Sidebar
    st.sidebar.title("Input Method")
    input_method = st.sidebar.radio(
        "Choose how to score a food item:",
        ["Manual Entry", "Barcode Lookup"],
        key="input_method",
    )
    
    # Main content
    if input_method == "Manual Entry":
        header_cols = st.columns([5, 1.5])
        with header_cols[0]:
            st.header("Manual Food Entry")
            st.caption(
                "Enter per-serving values from the nutrition label. "
                "The hints show the exact nutrient cutoffs used by this risk screen."
            )
        with header_cols[1]:
            if st.button("Reset Form", use_container_width=True):
                _reset_manual_form()
                _request_rerun()

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Basics")
            name = st.text_input("Food Name", value=MANUAL_DEFAULTS["name"], key="manual_name")
            brand = st.text_input("Brand (optional)", value=MANUAL_DEFAULTS["brand"], key="manual_brand")
            category = st.selectbox(
                "Category",
                MANUAL_CATEGORY_OPTIONS,
                index=0,
                key="manual_category",
            )
            serving_g = st.number_input(
                "Serving Size (g)",
                value=float(MANUAL_DEFAULTS["serving_g"]),
                min_value=1.0,
                key="manual_serving_g",
                help="Use the serving size shown on the label.",
            )
            calories = st.number_input(
                "Calories (kcal)",
                value=float(MANUAL_DEFAULTS["calories"]),
                min_value=0.0,
                key="manual_calories",
            )
            fat_g = st.number_input(
                "Fat (g)",
                value=float(MANUAL_DEFAULTS["fat_g"]),
                min_value=0.0,
                key="manual_fat_g",
            )

        with col2:
            st.markdown("#### Nutrition (Per Serving)")
            carbs_g = st.number_input(
                "Total Carbs (g)",
                value=float(MANUAL_DEFAULTS["carbs_g"]),
                min_value=0.0,
                key="manual_carbs_g",
                help="High-risk rule trigger: >= 30g per serving.",
            )
            fiber_g = st.number_input(
                "Fiber (g)",
                value=float(MANUAL_DEFAULTS["fiber_g"]),
                min_value=0.0,
                key="manual_fiber_g",
                help="Protective rule trigger: >= 5.6g per serving.",
            )
            sugar_g = st.number_input(
                "Total Sugar (g)",
                value=float(MANUAL_DEFAULTS["sugar_g"]),
                min_value=0.0,
                key="manual_sugar_g",
            )
            added_sugar_g = st.number_input(
                "Added Sugar (g)",
                value=float(MANUAL_DEFAULTS["added_sugar_g"]),
                min_value=0.0,
                key="manual_added_sugar_g",
                help="High-risk rule trigger: >= 10g per serving.",
            )
            protein_g = st.number_input(
                "Protein (g)",
                value=float(MANUAL_DEFAULTS["protein_g"]),
                min_value=0.0,
                key="manual_protein_g",
                help="Protective rule trigger: >= 10g per serving.",
            )
            sodium_mg = st.number_input(
                "Sodium (mg)",
                value=float(MANUAL_DEFAULTS["sodium_mg"]),
                min_value=0.0,
                key="manual_sodium_mg",
                help="High-risk rule trigger: >= 460mg per serving.",
            )

        if st.button("Score This Food", type="primary"):
            item = {
                "name": name,
                "brand": brand if brand else None,
                "category": category,
                "serving_g": serving_g,
                "calories": calories,
                "carbs_g": carbs_g,
                "fiber_g": fiber_g,
                "sugar_g": sugar_g,
                "added_sugar_g": added_sugar_g,
                "sugar_alcohols_g": 0.0,
                "protein_g": protein_g,
                "fat_g": fat_g,
                "sodium_mg": sodium_mg,
            }
            
            with st.spinner("Analyzing..."):
                result = score_item(item, model, df_candidates)
            st.session_state["last_result"] = result
    
    else:  # Barcode Lookup
        st.header("Barcode Lookup")
        st.markdown("*Enter a UPC/EAN barcode to fetch nutrition data from Open Food Facts.*")

        if "barcode_lookup_input" not in st.session_state:
            st.session_state["barcode_lookup_input"] = ""

        quick_cols = st.columns(2)
        with quick_cols[0]:
            st.caption("Soda")
            st.code("049000028904")
        with quick_cols[1]:
            st.caption("Bagels")
            st.code("5000436049135")

        barcode = st.text_input(
            "Barcode",
            key="barcode_lookup_input",
            placeholder="e.g., 078742101347",
            help="8, 12, 13, or 14 digit barcode"
        )

        lookup_clicked = st.button("Lookup & Score", type="primary")
        if lookup_clicked:
            if not barcode or not barcode.isdigit():
                st.error("Please enter a valid numeric barcode.")
            else:
                with st.spinner("Fetching product data from Open Food Facts..."):
                    result = score_by_barcode(barcode, model, df_candidates)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Show info about alternatives fetching
                    if result.get("alternatives_source") == "dynamic":
                        st.info(f"✅ Fetched {result.get('alternatives_count', 0)} real alternatives from OpenFoodFacts")
                    st.session_state["last_result"] = result
    
    if st.session_state.get("last_result"):
        render_result_columns(st.session_state["last_result"])
    else:
        display_alternatives_placeholder()

    with st.expander("About This Model"):
        st.caption(
            f"Config: `{_APP_CFG_ARGS.config}`"
            + (f" | Profile: `{_APP_CFG_ARGS.profile}`" if _APP_CFG_ARGS.profile else "")
        )
        if model_type == "unavailable":
            st.error("Active model type could not be determined from artifacts.")
        else:
            st.caption(f"Active model type: `{model_type}`")
        if metrics_timestamp:
            st.caption(f"Latest metrics timestamp (UTC): `{metrics_timestamp}`")
            if metrics_timestamp_source:
                st.caption(f"Timestamp source: `{metrics_timestamp_source}`")
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**Validation**")
                st.json(metrics.get("validation", {}))
            with col2:
                st.markdown("**Test**")
                st.json(metrics.get("test", {}))
            with col3:
                st.markdown("**Cross-Validation**")
                st.json(metrics.get("cv", {}))
            with col4:
                st.markdown("**Recommendation Eval**")
                if recommendation_eval:
                    st.json(recommendation_eval)
                else:
                    st.caption("No recommendation eval found in model metadata.")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Dianalysis v1.1.0</strong></p>
        <p>Educational demo for diabetes-aware food scoring.</p>
        <p>This tool provides educational guidance only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
