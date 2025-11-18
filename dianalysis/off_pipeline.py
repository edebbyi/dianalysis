"""
Open Food Facts API integration and data pipeline for fetching real product data.
"""

import re
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configuration
OFF_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"

# Display rules (units + trace thresholds)
DISPLAY_RULES = {
    "carbs_g": {"label": "Carbs", "bases": ["carbohydrates"], "unit": "g", "lt": 1},
    "fiber_g": {"label": "Fiber", "bases": ["fiber", "fibre"], "unit": "g", "lt": 1},
    "sugar_g": {"label": "Total sugar", "bases": ["sugars"], "unit": "g", "lt": 1},
    "added_sugar_g": {"label": "Added sugar", "bases": ["added-sugars", "added_sugars"], "unit": "g", "lt": 1},
    "sugar_alcohols_g": {"label": "Sugar alcohols", "bases": ["polyols", "sugar-alcohols", "sugar_alcohols"], "unit": "g", "lt": 1},
    "protein_g": {"label": "Protein", "bases": ["proteins"], "unit": "g", "lt": 1},
    "fat_g": {"label": "Fat", "bases": ["fat"], "unit": "g", "lt": 1},
    "sodium_mg": {"label": "Sodium", "bases": ["sodium"], "unit": "mg", "lt": 5},
}

# Multi-tag search space per alt_group
OFF_TAGS_MULTI = {
    "oats": ["oats"],
    "rice": ["rice"],
    "quinoa": ["quinoa"],
    "pasta-noodles": ["pasta", "noodles", "spaghetti", "penne", "ramen", "udon", "soba", "macaroni"],
    "cereal": ["breakfast-cereals", "cereals"],
    "nuts-seeds": ["nuts", "nuts-and-seeds", "seeds", "almonds", "cashews", "peanuts", "pistachios", "walnuts", "hazelnuts", "pecans", "macadamia", "trail-mix"],
    "snack": ["snacks", "chips", "crisps", "tortilla-chips", "pretzel"],
    "ice-cream": ["ice-creams", "ice-cream", "frozen-desserts"],
    "bread": ["breads", "bread", "bakery", "bagels", "tortillas", "flatbreads", "wraps", "buns", "rolls", "pita", "naan", "ciabatta"],
    "drink": ["beverages", "drinks", "soft-drinks", "juices", "water"],
}

# Canonical category for each alt_group
CANON_CATEGORY_FOR_GROUP = {
    "oats": "grain", "rice": "grain", "quinoa": "grain", "pasta-noodles": "grain",
    "cereal": "cereal", "granola": "cereal",
    "nuts-seeds": "nut", "snack": "snack", "bread": "bread", "ice-cream": "dessert", "drink": "drink",
}

# High-confidence positive name patterns for each alt_group
ALT_KEYWORDS = {
    "oats": [r"\boat(s|meal)?\b", r"\bgranola\b", r"\bmuesli\b", r"\bporridge\b"],
    "rice": [r"\brice\b", r"\bbasmati\b", r"\bjasmine\b", r"\bwild[- ]?rice\b"],
    "quinoa": [r"\bquinoa\b"],
    "pasta-noodles": [r"\bpasta\b", r"\bnoodle(s)?\b", r"\bspaghetti\b", r"\bpenne\b", r"\bmacaroni\b", r"\bramen\b", r"\budon\b", r"\bsoba\b"],
    "cereal": [r"\bcereal(s)?\b"],
    "nuts-seeds": [r"\bnut(s)?\b", r"\bseed(s)?\b", r"\balmond(s)?\b", r"\bcashew(s)?\b", r"\bpeanut(s)?\b",
                   r"\bpistachio(s)?\b", r"\bwalnut(s)?\b", r"\bhazelnut(s)?\b", r"\bpecan(s)?\b", r"\bmacadamia\b",
                   r"\btrail[- ]?mix\b", r"\bsunflower\b", r"\bpumpkin\b", r"\bchia\b", r"\bflax\b"],
    "bread": [r"\bbread(s)?\b", r"\bbagel(s)?\b", r"\btortilla(s)?\b", r"\bflatbread(s)?\b", r"\bwrap(s)?\b",
              r"\bbun(s)?\b", r"\broll(s)?\b", r"\bpita\b", r"\bnaan\b", r"\bciabatta\b"],
    "snack": [r"\bchips?\b", r"\bcrisps?\b", r"\bnacho(s)?\b", r"\btortilla chips\b", r"\bsnack\b", r"\bpretzel(s)?\b"],
    "ice-cream": [r"\bice[- ]?cream\b", r"\bfrozen dessert\b"],
    "drink": [r"\bbeverage(s)?\b", r"\bdrink(s)?\b", r"\bwater\b", r"\bjuice(s)?\b", r"\bsoft[- ]?drink(s)?\b", r"\bsoda(s)?\b"],
}

# Negative keywords to filter out wrong items
NEGATIVE_KEYWORDS = {
    "nuts-seeds": [r"\b(curry|soup|sauce|beans?|lentils?|chili|ready[- ]?meal|microwave)\b", r"\bwater\b", r"\bdrink(s)?\b"],
    "bread": [r"\bsoup\b", r"\bnoodle(s)?\b", r"\brice\b", r"\bjuice\b"],
    "oats": [r"\bnoodle(s)?\b", r"\bjuice\b", r"\bsoup\b", r"\brice\b"],
    "rice": [r"\bnoodle(s)?\b", r"\boat(s|meal)?\b", r"\bjuice\b"],
    "quinoa": [r"\bsoup\b", r"\bjuice\b"],
    "pasta-noodles": [r"\brice\b", r"\bjuice\b", r"\bsoup\b"],
    "cereal": [r"\bsoup\b", r"\bnoodle(s)?\b", r"\brice\b"],
    "ice-cream": [r"\bsoup\b", r"\bwater\b", r"\bnoodle(s)?\b"],
    "drink": [r"\bsoup\b", r"\bnoodle(s)?\b", r"\bpasta\b", r"\brice\b", r"\bice[- ]?cream\b"],
}

TARGET_GROUPS = set(OFF_TAGS_MULTI.keys())


# Helper functions
def safe_lower(x) -> str:
    """Return lowercase string or empty string for None/NaN/non-strings."""
    try:
        if isinstance(x, str):
            return x.lower()
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return str(x).lower()
    except Exception:
        return ""


def text_has_any(patterns, *texts) -> bool:
    """Check if any pattern matches in the combined text."""
    joined = " ".join([safe_lower(t) for t in texts if safe_lower(t)])
    return any(re.search(p, joined) for p in patterns)


def safe_float(x, default=0.0):
    """Safely convert to float with default."""
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def compute_net_carbs_local(row: dict) -> float:
    """Compute net carbs from a dictionary."""
    carbs = max(safe_float(row.get("carbs_g"), 0.0), 0.0)
    fiber = max(safe_float(row.get("fiber_g"), 0.0), 0.0)
    sugar_alc = max(safe_float(row.get("sugar_alcohols_g"), 0.0), 0.0)
    return max(carbs - fiber - sugar_alc, 0.0)


# HTTP session with retries
def make_session():
    """Create robust HTTP session with retries."""
    s = requests.Session()
    retries = Retry(
        total=5, connect=4, read=4, backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "dianalysis/0.1 (educational demo)"})
    return s


SESSION = make_session()


def safe_get(url, params=None, timeout=30):
    """Make HTTP GET request with session."""
    return SESSION.get(url, params=params, timeout=timeout)


# OFF data extraction helpers
def extract_categories(product):
    """Extract and clean category tags from OFF product."""
    raw = product.get("categories_hierarchy") or product.get("categories_tags") or []
    cats = []
    for c in raw:
        try:
            s = str(c).strip().lower()
            if not s:
                continue
            if ":" in s:
                s = s.split(":")[-1]
            cats.append(s)
        except Exception:
            continue
    return cats


def parse_serving_g(product):
    """Parse serving size from product data."""
    ss = (product.get("serving_size") or "").lower()
    m = re.search(r"([\d\.]+)\s*(g|ml)\b", ss)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        return val if unit == "g" else val  # ml≈g for liquids
    q = product.get("serving_quantity")
    u = (product.get("serving_unit") or "").lower()
    if q is not None and u == "g":
        return float(q)
    return 100.0


def get_nutrient(nutriments, base, serving_g, default=None):
    """Extract nutrient value from OFF nutriments dict."""
    for key in (f"{base}_serving", f"{base}_per_serving"):
        if key in nutriments and nutriments[key] is not None:
            try:
                return float(nutriments[key])
            except:
                pass
    v100 = nutriments.get(f"{base}_100g")
    if v100 is not None:
        try:
            return float(v100) * (serving_g / 100.0)
        except:
            pass
    v = nutriments.get(base)
    if v is not None:
        try:
            return float(v) * (serving_g / 100.0)
        except:
            pass
    return default


def get_first_nutrient(nutriments, bases, serving_g, default=None):
    """Try multiple base names to find nutrient."""
    for b in bases:
        v = get_nutrient(nutriments, b, serving_g, None)
        if v is not None:
            try:
                return float(v)
            except:
                pass
    return default


def get_calories(nutriments, serving_g):
    """Extract calories from nutriments."""
    kcal = get_first_nutrient(nutriments, ["energy-kcal", "energy_kcal"], serving_g, None)
    if kcal is not None:
        return float(kcal)
    kj = get_first_nutrient(nutriments, ["energy"], serving_g, None)
    return float(kj) * 0.239006 if kj is not None else 0.0


def get_sodium_mg(nutriments, serving_g):
    """Extract sodium in mg from nutriments."""
    unit = (nutriments or {}).get("sodium_unit", "g")
    if (nutriments or {}).get("sodium_serving") is not None:
        val = float(nutriments["sodium_serving"])
        return val * 1000 if unit == "g" else val
    if (nutriments or {}).get("sodium_100g") is not None:
        val = float(nutriments["sodium_100g"]) * (serving_g / 100.0)
        return val * 1000 if unit == "g" else val
    salt = get_nutrient(nutriments, "salt", serving_g, None)
    if salt is not None:
        return float(salt) * 0.393 * 1000
    return 0.0


def per100_to_serving(val_100g: float, serving_g: float) -> float:
    """Convert per-100g value to per-serving."""
    return float(val_100g) * (serving_g / 100.0)


def display_value(nutriments, bases, serving_g, serving_value, unit, lt_threshold):
    """Format nutrient value for display with trace handling."""
    v100 = None
    for b in bases:
        k = f"{b}_100g"
        if k in nutriments and nutriments[k] is not None:
            try:
                v100 = float(nutriments[k])
                break
            except:
                pass
    
    est_serv = None
    try:
        if v100 is not None and serving_g:
            if unit == "mg" and any(b == "sodium" for b in bases):
                unit_tag = nutriments.get("sodium_unit", "g")
                est = per100_to_serving(v100, serving_g)
                est_serv = est * 1000.0 if unit_tag == "g" else est
            else:
                est_serv = per100_to_serving(v100, serving_g)
    except:
        est_serv = None

    if serving_value is None:
        if est_serv is not None and 0 < est_serv < lt_threshold:
            return f"<{int(lt_threshold)}{unit}"
        return "not listed"
    
    try:
        v = float(serving_value)
    except:
        return "not listed"
    
    if v == 0.0:
        if est_serv is not None and 0 < est_serv < lt_threshold:
            return f"<{int(lt_threshold)}{unit}"
        return f"0{unit}"
    
    if unit == "mg":
        return f"{int(round(v))}mg"
    if unit == "kcal":
        return f"{int(round(v))}kcal"
    return f"{v:.1f}g"


# Category and group mapping
def map_category_and_group(product):
    """Map OFF product to (category, alt_group) tuple."""
    cats = extract_categories(product)
    
    def has(*subs):
        return any(any(s in c for s in subs) for c in cats)

    if has("oat", "oats", "porridge", "rolled-oats", "oatmeal", "granola", "muesli"):
        return ("grain", "oats")
    if has("rice", "brown-rice", "white-rice", "basmati", "jasmine", "wild-rice"):
        return ("grain", "rice")
    if has("quinoa"):
        return ("grain", "quinoa")
    if has("pasta", "noodles", "spaghetti", "penne", "macaroni", "ramen", "udon", "soba"):
        return ("grain", "pasta-noodles")

    if has("chips", "chip", "crisps", "nacho", "nachos", "tortilla-chips", "tortilla chips"):
        return ("snack", "snack")

    if has("bread", "breads", "bakery", "loaves", "bagel", "bagels",
           "flatbread", "flatbreads", "wrap", "wraps",
           "bun", "buns", "roll", "rolls", "pita", "naan", "ciabatta"):
        return ("bread", "bread")

    if has("breakfast-cereals", "cereals"):
        if has("granola", "muesli"):
            return ("cereal", "granola")
        return ("cereal", "cereal")

    if has("beverages", "drinks", "soft-drinks", "sodas", "juice", "juices", "water"):
        return ("drink", "drink")

    if has("ice-cream", "ice-creams", "frozen-dessert", "frozen-desserts"):
        return ("dessert", "ice-cream")
    
    if has("dairies", "dairy", "milk", "yogurt", "cheese", "cream"):
        return ("dairy", "dairy")

    if has("nuts", "nuts-and-seeds", "almonds", "cashews", "peanuts", "pistachios", "walnuts", "hazelnuts",
           "pecans", "macadamia", "seeds", "sunflower-seeds", "pumpkin-seeds", "chia", "flax",
           "trail-mix", "nut-mix", "seed-mix"):
        return ("nut", "nuts-seeds")

    if has("snacks", "chips", "crisps", "crackers", "popcorn", "bars"):
        return ("snack", "snack")
    
    return ("snack", "snack")


def fallback_group_from_text(name: str, ingredients: str, cats_list) -> tuple:
    """Infer (category, alt_group) from strong textual cues."""
    name_l = safe_lower(name)
    cats_l = [safe_lower(c) for c in (cats_list or [])]

    # Scan in priority order
    for ag in ["nuts-seeds", "snack", "bread", "oats", "rice", "quinoa", "pasta-noodles", "cereal", "ice-cream", "drink"]:
        if text_has_any(ALT_KEYWORDS.get(ag, []), name_l) or any(t in " ".join(cats_l) for t in OFF_TAGS_MULTI[ag]):
            return (CANON_CATEGORY_FOR_GROUP.get(ag, "snack"), ag)
    
    # Last resort: drinks by name
    if re.search(r"\b(water|juice|beverage|drink|soda)\b", name_l):
        return ("drink", "drink")
    
    return None


# Barcode validation
VALID_CODE_LEN = {8, 12, 13, 14}


def looks_like_barcode(code: str) -> bool:
    """Check if string looks like a valid barcode."""
    c = (code or "").strip()
    return c.isdigit() and (len(c) in VALID_CODE_LEN)


# Main fetch function
def fetch_and_normalize_off(barcode: str) -> dict:
    """
    Fetch and normalize a product from Open Food Facts.
    
    Args:
        barcode: UPC/EAN barcode string
        
    Returns:
        Dictionary with normalized nutrition data
        
    Raises:
        ValueError: If barcode is invalid or product not found
        requests.RequestException: If API call fails
    """
    if not looks_like_barcode(barcode):
        raise ValueError("bad barcode")
    
    r = safe_get(OFF_URL.format(barcode=barcode), timeout=20)
    r.raise_for_status()
    data = r.json()
    
    if data.get("status") != 1:
        raise ValueError("Product not found")

    p = data["product"]
    n = p.get("nutriments", {}) or {}
    sg = parse_serving_g(p)
    category, alt_group = map_category_and_group(p)
    cats = extract_categories(p)

    features = {
        "name": p.get("product_name_en") or p.get("product_name") or barcode,
        "brand": (p.get("brands") or "").split(",")[0].strip() if p.get("brands") else None,
        "upc": barcode,
        "source": "openfoodfacts",
        "created_at": datetime.now(timezone.utc).isoformat(),

        "category": category,
        "alt_group": alt_group,
        "categories_all": "|".join(cats) if cats else None,
        "pnns2": p.get("pnns_groups_2") or p.get("pnns_groups_2_en") or None,
        "ingredients_tags": "|".join(p.get("ingredients_tags") or []) or None,

        "serving_g": float(sg),
        "calories": get_calories(n, sg),

        "carbs_g": get_nutrient(n, "carbohydrates", sg, None),
        "fiber_g": get_first_nutrient(n, ["fiber", "fibre"], sg, None),
        "sugar_g": get_nutrient(n, "sugars", sg, None),
        "added_sugar_g": get_first_nutrient(n, ["added-sugars", "added_sugars"], sg, None),
        "sugar_alcohols_g": get_first_nutrient(n, ["polyols", "sugar-alcohols", "sugar_alcohols"], sg, None),
        "protein_g": get_nutrient(n, "proteins", sg, None),
        "fat_g": get_nutrient(n, "fat", sg, None),
        "sodium_mg": get_sodium_mg(n, sg),

        "ingredients_text": p.get("ingredients_text_en") or p.get("ingredients_text") or None,
    }

    # Add display formatting
    disp = {}
    for field, rule in DISPLAY_RULES.items():
        disp[field] = display_value(
            nutriments=n, bases=rule["bases"], serving_g=sg,
            serving_value=features.get(field), unit=rule["unit"], lt_threshold=rule["lt"]
        )
    features["__display"] = disp
    features["net_carbs_g"] = compute_net_carbs_local(features)
    
    return features


def infer_alt_group_for_item(item: dict) -> dict:
    """
    Ensure the item has a useful (category, alt_group) from target groups.
    Uses OFF tags, name keywords, and fallback text inference.
    """
    out = dict(item)
    ag = safe_lower(out.get("alt_group"))
    name = out.get("name") or ""
    ingr = out.get("ingredients_text") or ""
    cats_all = out.get("categories_all") or ""
    cats = cats_all.split("|") if isinstance(cats_all, str) and cats_all else []

    # If already good, normalize category to canonical and return
    if ag in TARGET_GROUPS:
        if ag in CANON_CATEGORY_FOR_GROUP:
            out["category"] = CANON_CATEGORY_FOR_GROUP[ag]
        return out

    # Try to infer from name/ingredients/tags
    fix = fallback_group_from_text(name, ingr, cats)
    if fix:
        new_cat, new_ag = fix
        out["alt_group"] = new_ag
        out["category"] = CANON_CATEGORY_FOR_GROUP.get(new_ag, new_cat)
        return out

    # As a final nudge, use name-only high-confidence keywords
    name_l = safe_lower(name)
    for guess_ag, patterns in ALT_KEYWORDS.items():
        if any(re.search(p, name_l) for p in patterns):
            out["alt_group"] = guess_ag
            out["category"] = CANON_CATEGORY_FOR_GROUP.get(guess_ag, out.get("category"))
            return out

    # If still unknown, leave as-is
    return out


def fetch_category_products(category: str, limit: int = 50) -> list:
    """
    Fetch products from OpenFoodFacts by category.
    
    Args:
        category: Category to search for (e.g., "nuts", "cereals", "bread")
        limit: Maximum number of products to fetch
        
    Returns:
        List of normalized product dictionaries
    """
    search_url = "https://world.openfoodfacts.org/cgi/search.pl"
    
    params = {
        "search_terms": category,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": min(limit, 100),  # API limit
        "page": 1,
        "fields": "code,product_name,product_name_en,brands,categories_tags,categories_hierarchy,pnns_groups_2,ingredients_tags,ingredients_text,ingredients_text_en,serving_size,serving_quantity,serving_unit,nutriments"
    }
    
    try:
        r = safe_get(search_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        products = []
        for product_data in data.get("products", [])[:limit]:
            try:
                # Extract barcode/code
                code = product_data.get("code") or product_data.get("_id", "")
                if not code or not looks_like_barcode(str(code)):
                    continue
                
                # Use the same normalization logic as fetch_and_normalize_off
                p = product_data
                n = p.get("nutriments", {}) or {}
                sg = parse_serving_g(p)
                category_mapped, alt_group = map_category_and_group(p)
                cats = extract_categories(p)
                
                features = {
                    "name": p.get("product_name_en") or p.get("product_name") or str(code),
                    "brand": (p.get("brands") or "").split(",")[0].strip() if p.get("brands") else None,
                    "upc": str(code),
                    "source": "openfoodfacts",
                    "created_at": datetime.now(timezone.utc).isoformat(),

                    "category": category_mapped,
                    "alt_group": alt_group,
                    "categories_all": "|".join(cats) if cats else None,
                    "pnns2": p.get("pnns_groups_2") or p.get("pnns_groups_2_en") or None,
                    "ingredients_tags": "|".join(p.get("ingredients_tags") or []) or None,

                    "serving_g": float(sg),
                    "calories": get_calories(n, sg),

                    "carbs_g": get_nutrient(n, "carbohydrates", sg, None),
                    "fiber_g": get_first_nutrient(n, ["fiber", "fibre"], sg, None),
                    "sugar_g": get_nutrient(n, "sugars", sg, None),
                    "added_sugar_g": get_first_nutrient(n, ["added-sugars", "added_sugars"], sg, None),
                    "sugar_alcohols_g": get_first_nutrient(n, ["polyols", "sugar-alcohols", "sugar_alcohols"], sg, None),
                    "protein_g": get_nutrient(n, "proteins", sg, None),
                    "fat_g": get_nutrient(n, "fat", sg, None),
                    "sodium_mg": get_sodium_mg(n, sg),

                    "ingredients_text": p.get("ingredients_text_en") or p.get("ingredients_text") or None,
                }

                # Add display formatting
                disp = {}
                for field, rule in DISPLAY_RULES.items():
                    disp[field] = display_value(
                        nutriments=n, bases=rule["bases"], serving_g=sg,
                        serving_value=features.get(field), unit=rule["unit"], lt_threshold=rule["lt"]
                    )
                features["__display"] = disp
                features["net_carbs_g"] = compute_net_carbs_local(features)
                
                # Ensure we have valid nutritional data
                if features.get("carbs_g") is not None or features.get("calories", 0) > 0:
                    products.append(features)
                
                # Add small delay to be respectful to the API
                time.sleep(0.1)
                
            except Exception as e:
                # Skip problematic products
                continue
        
        return products
        
    except Exception as e:
        print(f"Error fetching category products for {category}: {e}")
        return []
