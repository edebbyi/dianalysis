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
    os.environ["DIANALYSIS_RETRIEVAL_BACKEND"] = str(cfg_get(_APP_CFG, "retrieval", "backend", default="heuristic"))
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
    from dianalysis.model import load_model, generate_synthetic_data, train_model
    from dianalysis.scoring import score_item, score_by_barcode
except ModuleNotFoundError as e:
    _MODEL_IMPORT_ERROR = e

ARTIFACTS_DIR = str(cfg_get(_APP_CFG, "paths", "artifacts_dir", default="artifacts"))
CLEAN_CSV_PATH = str(cfg_get(_APP_CFG, "paths", "input_csv", default="data/products_off_clean.csv"))
SCORED_CSV_PATH = str(cfg_get(_APP_CFG, "paths", "scored_csv", default="data/products_off_clean_scored.csv"))

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
    .risk-low { color: #28a745; font-weight: bold; font-size: 2em; }
    .risk-medium { color: #ffc107; font-weight: bold; font-size: 2em; }
    .risk-high { color: #dc3545; font-weight: bold; font-size: 2em; }
    .alternative-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        color: #333;
    }
    .alternative-card h4 {
        color: #333 !important;
        margin-bottom: 8px;
    }
    .alternative-card p {
        color: #555 !important;
        margin-bottom: 4px;
    }
    .tier-good { border-left: 4px solid #6c757d; }
    .tier-better { border-left: 4px solid #17a2b8; }
    .tier-best { border-left: 4px solid #28a745; }
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


def display_nutrition_table(display_dict: dict[str, Any]) -> None:
    """Display nutrition facts table."""
    st.subheader("📊 Nutrition Facts")
    
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


def display_alternatives(alternatives: list[dict[str, Any]]) -> None:
    """Display alternative recommendations."""
    if not alternatives:
        st.info("No alternatives found in the same food group.")
        return

    st.subheader("🔄 Better Alternatives")
    st.markdown("*These alternatives are in the same food group and have better nutritional profiles.*")

    tier_order = {"best": 0, "better": 1, "good": 2}
    ranked_alternatives = sorted(
        alternatives,
        key=lambda alt: tier_order.get(str(alt.get("tier", "good")).lower(), 99),
    )

    for alt in ranked_alternatives:
        tier = alt.get("tier", "Good")
        tier_class = f"tier-{tier.lower()}"
        
        st.markdown(f"""
        <div class="alternative-card {tier_class}">
            <h4>{tier} Choice: {alt.get('name', 'Unknown')}</h4>
            <p><strong>Brand:</strong> {alt.get('brand', 'N/A')}</p>
            <p><strong>Risk Score:</strong> {alt.get('risk_display', alt.get('risk_score', '—'))}</p>
            <p><strong>Why better:</strong> {alt.get('why', 'Lower risk in same category')}</p>
        </div>
        """, unsafe_allow_html=True)


def render_result_columns(result: dict[str, Any] | None) -> None:
    """Render item details on left and alternatives on right."""
    if not result:
        return
    
    risk_score = result.get("risk_score", 0)
    risk_class = get_risk_class(risk_score)

    st.divider()
    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown(f"""
        ## Risk Score
        <p class="{risk_class}">{result.get('risk_display', risk_score)}</p>
        """, unsafe_allow_html=True)
        
        st.subheader(f"{result.get('item_name', 'Unknown Food')}")
        if result.get('item_brand'):
            st.caption(f"Brand: {result['item_brand']}")
        st.caption(f"Category: {result.get('item_category', 'Unknown')} | Group: {result.get('item_alt_group', 'Unknown')}")
        
        st.subheader("📝 Why This Score?")
        for reason in result.get("reasons", []):
            st.write(f"• {reason}")
        
        display_nutrition_table(result.get("display", {}))

        if result.get("notes"):
            with st.expander("ℹ️ Data Notes"):
                for note in result["notes"]:
                    st.info(note)

    with cols[1]:
        display_alternatives(result.get("alternatives", []))


def display_alternatives_placeholder() -> None:
    cols = st.columns([3, 2])
    with cols[0]:
        st.write("")
    with cols[1]:
        st.subheader("🔄 Better Alternatives")
        st.info("Submit a food item to score it and see healthier swaps appear here.")


def main() -> None:
    """Main app function."""
    
    # Load model and data
    model, meta, metrics = load_trained_model(_artifact_state())
    df_candidates, candidate_messages = load_candidates_data()
    for style, text, key in candidate_messages:
        _show_dismissable_message(key, text, style=style)
    
    # Header
    st.title("🍎 Dianalysis")
    st.markdown("### Diabetes-aware food scoring and recommendations")
    st.markdown(
        "Scan a food, get a clear diabetes-risk signal, and discover smarter same-category swaps in seconds."
    )
    st.markdown(
        "Your **risk score** is a calibrated 0‑100 probability (lower is better). "
        "Under the hood, Dianalysis combines nutrition-rule logic with a trained classifier "
        "(logistic or XGBoost, depending on loaded artifacts)."
    )
    st.caption(
        f"Config: `{_APP_CFG_ARGS.config}`"
        + (f" | Profile: `{_APP_CFG_ARGS.profile}`" if _APP_CFG_ARGS.profile else "")
    )
    
    recommendation_eval = meta.get("recommendation_eval", {}) if isinstance(meta, dict) else {}
    model_type = _infer_model_type(meta, ARTIFACTS_DIR)
    if isinstance(meta, dict):
        meta["model_type"] = model_type
    metrics_timestamp, metrics_timestamp_source = _metrics_timestamp_utc(meta, ARTIFACTS_DIR)

    if metrics:
        with st.expander("📈 Model Performance Metrics"):
            if model_type == "unavailable":
                st.error("Active model type could not be determined from artifacts.")
            else:
                st.caption(f"Active model type: `{model_type}`")
            if metrics_timestamp:
                st.caption(f"Latest metrics timestamp (UTC): `{metrics_timestamp}`")
                if metrics_timestamp_source:
                    st.caption(f"Timestamp source: `{metrics_timestamp_source}`")
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
    
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    # Sidebar
    st.sidebar.title("Input Method")
    st.sidebar.caption(
        f"Config: `{_APP_CFG_ARGS.config}`"
        + (f"\nProfile: `{_APP_CFG_ARGS.profile}`" if _APP_CFG_ARGS.profile else "")
    )
    input_method = st.sidebar.radio(
        "Choose how to score a food item:",
        ["Manual Entry", "Barcode Lookup"]
    )
    
    # Main content
    if input_method == "Manual Entry":
        st.header("Manual Food Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Food Name", value="Frosted Cereal")
            brand = st.text_input("Brand (optional)", value="")
            category = st.selectbox(
                "Category",
                ["cereal", "bread", "snack", "drink", "dairy", "grain"],
                index=0
            )
            serving_g = st.number_input("Serving Size (g)", value=40.0, min_value=1.0)
            calories = st.number_input("Calories", value=160.0, min_value=0.0)
        
        with col2:
            carbs_g = st.number_input("Carbs (g)", value=37.0, min_value=0.0)
            fiber_g = st.number_input("Fiber (g)", value=3.0, min_value=0.0)
            sugar_g = st.number_input("Total Sugar (g)", value=14.0, min_value=0.0)
            added_sugar_g = st.number_input("Added Sugar (g)", value=10.0, min_value=0.0)
            protein_g = st.number_input("Protein (g)", value=3.0, min_value=0.0)
            fat_g = st.number_input("Fat (g)", value=2.0, min_value=0.0)
            sodium_mg = st.number_input("Sodium (mg)", value=240.0, min_value=0.0)
        
        if st.button("🔍 Score This Food", type="primary"):
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
        
        barcode = st.text_input(
            "Barcode",
            value="",
            placeholder="e.g., 078742101347",
            help="8, 12, 13, or 14 digit barcode"
        )
        
        if st.button("🔍 Lookup & Score", type="primary"):
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

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Dianalysis v0.1.0</strong></p>
        <p>Educational demo for diabetes-aware food scoring.</p>
        <p>This tool provides educational guidance only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
