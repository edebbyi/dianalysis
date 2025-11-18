"""
Dianalysis: Diabetes-aware food scoring Streamlit app.
"""

import streamlit as st
import pandas as pd
import os
from dianalysis.model import load_model, generate_synthetic_data, train_model, compute_net_carbs
from dianalysis.scoring import score_item, score_by_barcode

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
    .tier-good { border-left: 4px solid #28a745; }
    .tier-better { border-left: 4px solid #17a2b8; }
    .tier-best { border-left: 4px solid #007bff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Load or train the model."""
    artifacts_dir = "artifacts"
    
    if os.path.exists(os.path.join(artifacts_dir, "model.joblib")):
        try:
            model, meta = load_model(artifacts_dir)
            return model, meta, None
        except Exception as e:
            st.warning(f"Could not load existing model: {e}. Training new model...")
    
    # Train new model
    with st.spinner("Training model on synthetic data..."):
        df = generate_synthetic_data(n=1000, random_state=42)
        model, metrics = train_model(df, artifacts_dir=artifacts_dir)
        meta = {"num_cols": None, "cat_cols": None, "categories": None}
    return model, meta, metrics


@st.cache_data
def load_candidates_data():
    """Load candidate products for alternatives from OpenFoodFacts."""
    clean_csv = "data/products_off_clean.csv"
    
    messages = []
    if os.path.exists(clean_csv):
        try:
            df = pd.read_csv(clean_csv, dtype={"upc": str})
            df = _append_curated_candidates(df)
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
            df = _append_curated_candidates(df)
            messages.append(("success", f"✅ Loaded {len(df)} real products from OpenFoodFacts", "off_fetch_success"))
            return df, messages
        else:
            st.warning("⚠️ Could not fetch real data, using synthetic alternatives")
    except Exception as e:
        st.warning(f"⚠️ OpenFoodFacts fetch failed: {e}, using synthetic alternatives")
    
    # Fallback: use synthetic data
    df = generate_synthetic_data(n=500, random_state=42)
    df = _append_curated_candidates(df)
    messages.append(("info", "📊 Using synthetic data for alternatives (demo mode)", "off_fetch_synthetic"))
    return df, messages


def _show_dismissable_message(key: str, text: str, style: str = "info"):
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


def _curated_snack_candidates():
    """Add curated snack/chip entries to the candidate pool."""
    now = pd.Timestamp.utcnow().isoformat()
    snacks = [
        {
            "name": "Baked Corn Chips",
            "brand": "Harvest Trail",
            "category": "snack",
            "alt_group": "snack",
            "serving_g": 30.0,
            "calories": 140.0,
            "carbs_g": 18.0,
            "fiber_g": 3.0,
            "sugar_g": 0.5,
            "added_sugar_g": 0.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 2.0,
            "fat_g": 5.5,
            "sodium_mg": 180.0,
            "ingredients_text": "corn flour, sunflower oil, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000111000111",
        },
        {
            "name": "Sea Salt Lentil Chips",
            "brand": "Crunch Herb",
            "category": "snack",
            "alt_group": "snack",
            "serving_g": 28.0,
            "calories": 130.0,
            "carbs_g": 15.0,
            "fiber_g": 4.5,
            "sugar_g": 1.0,
            "added_sugar_g": 0.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 4.0,
            "fat_g": 4.0,
            "sodium_mg": 170.0,
            "ingredients_text": "split lentils, rice flour, sunflower oil, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000222000222",
        },
        {
            "name": "Multigrain Thin Chips",
            "brand": "Wholesome Bites",
            "category": "snack",
            "alt_group": "snack",
            "serving_g": 28.0,
            "calories": 120.0,
            "carbs_g": 16.0,
            "fiber_g": 3.0,
            "sugar_g": 0.5,
            "added_sugar_g": 0.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 3.0,
            "fat_g": 3.5,
            "sodium_mg": 150.0,
            "ingredients_text": "whole grain flour blend, vegetable oils, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000333000333",
        },
        {
            "name": "Roasted Mixed Nuts",
            "brand": "Harvest Road",
            "category": "nut",
            "alt_group": "nuts-seeds",
            "serving_g": 30.0,
            "calories": 170.0,
            "carbs_g": 6.0,
            "fiber_g": 3.5,
            "sugar_g": 1.5,
            "added_sugar_g": 0.5,
            "sugar_alcohols_g": 0.0,
            "protein_g": 6.0,
            "fat_g": 14.0,
            "sodium_mg": 80.0,
            "ingredients_text": "almonds, cashews, walnuts, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000777000777",
        },
        {
            "name": "Curry Spiced Pistachios",
            "brand": "Spice Grove",
            "category": "nut",
            "alt_group": "nuts-seeds",
            "serving_g": 28.0,
            "calories": 160.0,
            "carbs_g": 5.0,
            "fiber_g": 3.0,
            "sugar_g": 1.0,
            "added_sugar_g": 0.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 6.0,
            "fat_g": 13.0,
            "sodium_mg": 120.0,
            "ingredients_text": "pistachios, curry powder, sunflower oil, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000888000888",
        },
        {
            "name": "Reduced Cocoa Sandwich Cookie",
            "brand": "Nutri Bakes",
            "category": "snack",
            "alt_group": "snack",
            "serving_g": 30.0,
            "calories": 160.0,
            "carbs_g": 22.0,
            "fiber_g": 4.0,
            "sugar_g": 9.0,
            "added_sugar_g": 6.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 3.0,
            "fat_g": 8.0,
            "sodium_mg": 180.0,
            "ingredients_text": "whole wheat flour, cocoa powder, chicory root fiber, vegetable oil",
            "source": "curated",
            "created_at": now,
            "upc": "000444000444",
        },
        {
            "name": "Almond Butter Oat Cookie",
            "brand": "Harvest Grain",
            "category": "snack",
            "alt_group": "snack",
            "serving_g": 25.0,
            "calories": 140.0,
            "carbs_g": 18.0,
            "fiber_g": 3.5,
            "sugar_g": 7.0,
            "added_sugar_g": 4.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 4.0,
            "fat_g": 6.0,
            "sodium_mg": 140.0,
            "ingredients_text": "oats, almond butter, flaxseed, honey",
            "source": "curated",
            "created_at": now,
            "upc": "000555000555",
        },
        {
            "name": "Chocolate Chip Crisp",
            "brand": "Wholesome Crunch",
            "category": "snack",
            "alt_group": "snack",
            "serving_g": 28.0,
            "calories": 150.0,
            "carbs_g": 20.0,
            "fiber_g": 3.0,
            "sugar_g": 10.0,
            "added_sugar_g": 7.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 2.0,
            "fat_g": 7.0,
            "sodium_mg": 120.0,
            "ingredients_text": "oat flour, cane sugar, chocolate chips, sunflower oil",
            "source": "curated",
            "created_at": now,
            "upc": "000666000666",
        },
    ]
    df = pd.DataFrame(snacks)
    df["net_carbs_g"] = df.apply(lambda row: compute_net_carbs(row), axis=1)
    return df


def _curated_nut_candidates():
    """Add healthier nut and seed candidates with extra fiber."""
    now = pd.Timestamp.utcnow().isoformat()
    nuts = [
        {
            "name": "High-Fiber Nut Medley",
            "brand": "Pulse Root",
            "category": "nut",
            "alt_group": "nuts-seeds",
            "serving_g": 30.0,
            "calories": 160.0,
            "carbs_g": 9.0,
            "fiber_g": 5.5,
            "sugar_g": 1.5,
            "added_sugar_g": 0.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 5.0,
            "fat_g": 12.0,
            "sodium_mg": 80.0,
            "ingredients_text": "almonds, walnuts, tiger nuts, flax seeds, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000999000999",
        },
        {
            "name": "Sprouted Pumpkin Seed Crisp",
            "brand": "Kernel Health",
            "category": "nut",
            "alt_group": "nuts-seeds",
            "serving_g": 30.0,
            "calories": 150.0,
            "carbs_g": 8.0,
            "fiber_g": 6.0,
            "sugar_g": 0.5,
            "added_sugar_g": 0.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 7.0,
            "fat_g": 11.0,
            "sodium_mg": 110.0,
            "ingredients_text": "sprouted pumpkin seeds, chickpea flour, olive oil, sea salt",
            "source": "curated",
            "created_at": now,
            "upc": "000888000888",
        },
        {
            "name": "Curated Pistachio Crunch",
            "brand": "Green Shell",
            "category": "nut",
            "alt_group": "nuts-seeds",
            "serving_g": 28.0,
            "calories": 170.0,
            "carbs_g": 7.0,
            "fiber_g": 6.5,
            "sugar_g": 2.0,
            "added_sugar_g": 1.0,
            "sugar_alcohols_g": 0.0,
            "protein_g": 6.0,
            "fat_g": 13.0,
            "sodium_mg": 100.0,
            "ingredients_text": "pistachios, sea salt, olive oil",
            "source": "curated",
            "created_at": now,
            "upc": "000777000777",
        },
    ]
    df = pd.DataFrame(nuts)
    df["net_carbs_g"] = df.apply(lambda row: compute_net_carbs(row), axis=1)
    return df


def _append_curated_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Append curated snack+nut candidates and dedupe."""
    curated = pd.concat(
        [
            _curated_snack_candidates(),
            _curated_nut_candidates(),
        ],
        ignore_index=True,
        sort=False,
    )
    df = pd.concat([df, curated], ignore_index=True, sort=False)
    return _drop_duplicate_candidates(df)


def _drop_duplicate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate candidate pool by name/brand to keep curated items unique."""
    if {"name", "brand"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["name", "brand"], keep="first")
    return df


def get_risk_class(score):
    """Get CSS class for risk score."""
    if score < 30:
        return "risk-low"
    elif score < 70:
        return "risk-medium"
    else:
        return "risk-high"


def display_nutrition_table(display_dict):
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


def display_alternatives(alternatives):
    """Display alternative recommendations."""
    if not alternatives:
        st.info("No alternatives found in the same food group.")
        return

    st.subheader("🔄 Better Alternatives")
    st.markdown("*These alternatives are in the same food group and have better nutritional profiles.*")
    
    for alt in alternatives:
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


def render_result_columns(result):
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


def display_alternatives_placeholder():
    cols = st.columns([3, 2])
    with cols[0]:
        st.write("")
    with cols[1]:
        st.subheader("🔄 Better Alternatives")
        st.info("Submit a food item to score it and see healthier swaps appear here.")


def main():
    """Main app function."""
    
    # Load model and data
    model, meta, metrics = load_trained_model()
    df_candidates, candidate_messages = load_candidates_data()
    for style, text, key in candidate_messages:
        _show_dismissable_message(key, text, style=style)
    
    # Header
    st.title("🍎 Dianalysis")
    st.markdown("### Diabetes-aware food scoring and recommendations")
    st.markdown(
        "Dianalysis is an educational demo that flags diabetes-relevant nutrition risks "
        "and suggests better swaps from the same food group."
    )
    st.markdown(
        "The **risk score** is a calibrated probability of being a high-risk product, "
        "scaled to 0‑100 (lower is better). "
        "It blends nutrition rules and a logistic model trained on synthetic food profiles."
    )
    
    if metrics:
        with st.expander("📈 Model Performance Metrics"):
            col1, col2 = st.columns(2)
            with col1:
                st.json(metrics.get("validation", {}))
            with col2:
                st.json(metrics.get("test", {}))
    
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    # Sidebar
    st.sidebar.title("Input Method")
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
