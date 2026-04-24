"""Fetch and clean Open Food Facts data for model and app use."""

from __future__ import annotations

import copy
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence, cast

import requests
import numpy as np
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .run_config import cfg_get, load_runtime_config
from .recommendation.candidate_pool import ensure_row_group_fields


# Configuration
_RUNTIME_CFG = load_runtime_config(
    Path(os.getenv("DIANALYSIS_CONFIG", "configs/base.toml")),
    (Path(os.environ["DIANALYSIS_PROFILE"]) if os.getenv("DIANALYSIS_PROFILE") else None),
)
OFF_BASE_URL = str(cfg_get(_RUNTIME_CFG, "off_api", "base_url", default="https://world.openfoodfacts.org")).rstrip("/")
OFF_REQUEST_TIMEOUT_SEC = int(cfg_get(_RUNTIME_CFG, "off_api", "request_timeout_sec", default=30))
OFF_RETRY_TOTAL = int(cfg_get(_RUNTIME_CFG, "off_api", "retry_total", default=5))
OFF_RETRY_BACKOFF = float(cfg_get(_RUNTIME_CFG, "off_api", "retry_backoff", default=0.7))
OFF_FALLBACK_BASE_URLS = tuple(
    dict.fromkeys(
        [
            OFF_BASE_URL,
            "https://world.openfoodfacts.org",
            "https://us.openfoodfacts.org",
        ]
    )
)
OFF_PRODUCT_API_PATHS = ("/api/v2/product/{barcode}.json", "/api/v0/product/{barcode}.json")

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
    "drink": ["beverages", "beverage", "drinks", "drink", "soft-drinks", "soft-drink", "sodas", "soda", "colas", "cola", "juices", "juice", "water"],
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
    "drink": [
        r"\bbeverage(s)?\b",
        r"\bdrink(s)?\b",
        r"\bwater\b",
        r"\bjuice(s)?\b",
        r"\bsoft[- ]?drink(s)?\b",
        r"\bsoda(s)?\b",
        r"\bcola(s)?\b",
        r"\bcoke\b",
        r"\bpop\b",
    ],
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
    "snack": [r"\b(water|beverage|drink|soda|cola|juice)\b"],
}

TARGET_GROUPS = set(OFF_TAGS_MULTI.keys())


# Helper functions
def safe_lower(x: Any) -> str:
    """Return a lowercase string, or an empty string for missing values."""
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


def text_has_any(patterns: Sequence[str], *texts: Any) -> bool:
    """Return True when any regex pattern matches the joined input text."""
    joined = " ".join([safe_lower(t) for t in texts if safe_lower(t)])
    return any(re.search(p, joined) for p in patterns)


def safe_float(x: Any, default: float = 0.0) -> float:
    """Convert a value to float, or return the default value if conversion fails."""
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def compute_net_carbs_local(row: dict) -> float:
    """Calculate net carbs as carbs - fiber - sugar alcohols, clipped at zero."""
    carbs = max(safe_float(row.get("carbs_g"), 0.0), 0.0)
    fiber = max(safe_float(row.get("fiber_g"), 0.0), 0.0)
    sugar_alc = max(safe_float(row.get("sugar_alcohols_g"), 0.0), 0.0)
    return max(carbs - fiber - sugar_alc, 0.0)


# HTTP session with retries
def make_session() -> requests.Session:
    """Create an HTTP session with retry and backoff settings."""
    s = requests.Session()
    connect_retry = max(0, OFF_RETRY_TOTAL - 1)
    retries = Retry(
        total=OFF_RETRY_TOTAL,
        connect=connect_retry,
        read=connect_retry,
        backoff_factor=OFF_RETRY_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "dianalysis/0.1 (educational demo)"})
    return s


SESSION = make_session()


def safe_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> requests.Response:
    """Send a GET request through the shared retry-enabled session."""
    return SESSION.get(url, params=params, timeout=(timeout or OFF_REQUEST_TIMEOUT_SEC))


# OFF data extraction helpers
def extract_categories(product: dict[str, Any]) -> list[str]:
    """Read category tags from an OFF product and normalize them to lowercase."""
    raw = product.get("categories_hierarchy") or product.get("categories_tags") or []
    raw_list: list[str]
    if isinstance(raw, str):
        raw_list = [x.strip() for x in raw.split(",") if x and x.strip()]
    elif isinstance(raw, (list, tuple, set)):
        raw_list = [str(x) for x in raw]
    else:
        raw_list = []

    # Some OFF responses only include plain category text.
    if not raw_list:
        fallback_text = product.get("categories") or product.get("categories_en") or ""
        if isinstance(fallback_text, str):
            raw_list = [x.strip() for x in fallback_text.split(",") if x and x.strip()]

    cats = []
    for c in raw_list:
        try:
            s = str(c).strip().lower()
            if not s:
                continue
            if ":" in s:
                s = s.split(":")[-1]
            cats.append(s)
            # Normalize common separators so matching catches "soft drinks" and "soft_drinks".
            if " " in s:
                cats.append(s.replace(" ", "-"))
            if "_" in s:
                cats.append(s.replace("_", "-"))
        except Exception:
            continue
    return cats


def parse_serving_g(product: dict[str, Any]) -> float:
    """Parse serving size and return grams per serving."""
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


def get_nutrient(
    nutriments: dict[str, Any],
    base: str,
    serving_g: float,
    default: float | None = None,
) -> float | None:
    """Read one nutrient from OFF nutriments using serving-first, then 100g fields."""
    for key in (f"{base}_serving", f"{base}_per_serving"):
        if key in nutriments and nutriments[key] is not None:
            try:
                return float(nutriments[key])
            except Exception:
                pass
    v100 = nutriments.get(f"{base}_100g")
    if v100 is not None:
        try:
            return float(v100) * (serving_g / 100.0)
        except Exception:
            pass
    v = nutriments.get(base)
    if v is not None:
        try:
            return float(v) * (serving_g / 100.0)
        except Exception:
            pass
    return default


def get_first_nutrient(
    nutriments: dict[str, Any],
    bases: Sequence[str],
    serving_g: float,
    default: float | None = None,
) -> float | None:
    """Try several nutrient key names and return the first valid value."""
    for b in bases:
        v = get_nutrient(nutriments, b, serving_g, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default


def get_calories(nutriments: dict[str, Any], serving_g: float) -> float:
    """Return calories per serving, converting from kJ when needed."""
    kcal = get_first_nutrient(nutriments, ["energy-kcal", "energy_kcal"], serving_g, None)
    if kcal is not None:
        return float(kcal)
    kj = get_first_nutrient(nutriments, ["energy"], serving_g, None)
    return float(kj) * 0.239006 if kj is not None else 0.0


def get_sodium_mg(nutriments: dict[str, Any], serving_g: float) -> float | None:
    """Return sodium per serving in milligrams, or None when unavailable."""
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
    return None


def per100_to_serving(val_100g: float, serving_g: float) -> float:
    """Convert a per-100g nutrient value into a per-serving value."""
    return float(val_100g) * (serving_g / 100.0)


def display_value(
    nutriments: dict[str, Any],
    bases: Sequence[str],
    serving_g: float,
    serving_value: Any,
    unit: str,
    lt_threshold: float,
) -> str:
    """Format a nutrient for UI display, including trace-level '<x' handling."""
    v100 = None
    for b in bases:
        k = f"{b}_100g"
        if k in nutriments and nutriments[k] is not None:
            try:
                v100 = float(nutriments[k])
                break
            except Exception:
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
    except Exception:
        est_serv = None

    if serving_value is None:
        if est_serv is not None and 0 < est_serv < lt_threshold:
            return f"<{int(lt_threshold)}{unit}"
        return "not listed"
    
    try:
        v = float(serving_value)
    except Exception:
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
def map_category_and_group(product: dict[str, Any]) -> tuple[str, str]:
    """Map an OFF product to the app's `(category, alt_group)` values."""
    cats = extract_categories(product)
    pnns2 = safe_lower(product.get("pnns_groups_2") or product.get("pnns_groups_2_en") or "")
    name_l = safe_lower(product.get("product_name_en") or product.get("product_name") or "")
    
    def has(*subs: str) -> bool:
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

    if has("beverages", "beverage", "drinks", "drink", "soft-drinks", "soft-drink", "sodas", "soda", "colas", "cola", "juice", "juices", "water"):
        return ("drink", "drink")
    if ("beverage" in pnns2) or text_has_any(ALT_KEYWORDS["drink"], name_l):
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


def fallback_group_from_text(
    name: str, ingredients: str, cats_list: Sequence[str] | None
) -> tuple[str, str] | None:
    """Guess `(category, alt_group)` from product name, ingredients, and category text."""
    name_l = safe_lower(name)
    cats_l = [safe_lower(c) for c in (cats_list or [])]

    # Scan in priority order
    for ag in ["drink", "nuts-seeds", "snack", "bread", "oats", "rice", "quinoa", "pasta-noodles", "cereal", "ice-cream"]:
        if text_has_any(ALT_KEYWORDS.get(ag, []), name_l) or any(t in " ".join(cats_l) for t in OFF_TAGS_MULTI[ag]):
            return (CANON_CATEGORY_FOR_GROUP.get(ag, "snack"), ag)
    
    # Last resort: drinks by name
    if re.search(r"\b(water|juice|beverage|drink|soda|cola|coke|pop)\b", name_l):
        return ("drink", "drink")
    
    return None


# Barcode validation
VALID_CODE_LEN = {8, 12, 13, 14}


def looks_like_barcode(code: str) -> bool:
    """Return True when the input looks like a valid barcode format."""
    c = (code or "").strip()
    return c.isdigit() and (len(c) in VALID_CODE_LEN)


# Main fetch function
@lru_cache(maxsize=512)
def _fetch_and_normalize_off_cached(barcode: str) -> dict[str, Any]:
    """Cached OFF barcode normalization to avoid repeated network fetches for the same code."""
    if not looks_like_barcode(barcode):
        raise ValueError("bad barcode")

    barcode_candidates = [barcode]
    stripped = barcode.lstrip("0")
    if stripped and stripped != barcode and len(stripped) in VALID_CODE_LEN:
        barcode_candidates.append(stripped)

    data: dict[str, Any] | None = None
    matched_barcode = barcode

    for code in barcode_candidates:
        for base_url in OFF_FALLBACK_BASE_URLS:
            for path in OFF_PRODUCT_API_PATHS:
                url = f"{base_url}{path.format(barcode=code)}"
                r = safe_get(url)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                payload = r.json()
                if payload.get("status") == 1 and isinstance(payload.get("product"), dict):
                    data = payload
                    matched_barcode = code
                    break
            if data is not None:
                break
        if data is not None:
            break

    if data is None:
        raise ValueError(
            f"Product not found for barcode {barcode} in Open Food Facts. "
            "Try another barcode or use Manual Entry."
        )

    p = data["product"]
    n = p.get("nutriments", {}) or {}
    sg = parse_serving_g(p)
    category, alt_group = map_category_and_group(p)
    cats = extract_categories(p)

    features = {
        "name": p.get("product_name_en") or p.get("product_name") or barcode,
        "brand": (p.get("brands") or "").split(",")[0].strip() if p.get("brands") else None,
        "upc": matched_barcode,
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
        rule_bases = cast(Sequence[str], rule["bases"])
        rule_unit = cast(str, rule["unit"])
        rule_lt = float(cast(int | float, rule["lt"]))
        disp[field] = display_value(
            nutriments=n, bases=rule_bases, serving_g=sg,
            serving_value=features.get(field), unit=rule_unit, lt_threshold=rule_lt
        )
    features["__display"] = disp
    features["net_carbs_g"] = compute_net_carbs_local(features)
    
    return ensure_row_group_fields(features)


def fetch_and_normalize_off(barcode: str) -> dict[str, Any]:
    """
    Fetch one OFF product by barcode and return normalized features.

    Uses an in-process cache so repeated lookups of the same barcode are fast.
    """
    normalized = _fetch_and_normalize_off_cached((barcode or "").strip())
    # Return a copy so callers can safely mutate fields without polluting cache.
    return copy.deepcopy(normalized)


def infer_alt_group_for_item(item: dict[str, Any]) -> dict[str, Any]:
    """Fill missing or weak category/group labels using tags and keyword-based fallbacks."""
    out = dict(item)
    ag = safe_lower(out.get("alt_group"))
    name = out.get("name") or ""
    ingr = out.get("ingredients_text") or ""
    cats_all = out.get("categories_all") or ""
    cats = cats_all.split("|") if isinstance(cats_all, str) and cats_all else []

    # If already good, normalize category to canonical and return.
    # Special case: allow a correction away from generic "snack" when text strongly indicates another group.
    if ag in TARGET_GROUPS:
        if ag == "snack":
            fix = fallback_group_from_text(name, ingr, cats)
            if fix and fix[1] != "snack":
                new_cat, new_ag = fix
                out["alt_group"] = new_ag
                out["category"] = CANON_CATEGORY_FOR_GROUP.get(new_ag, new_cat)
                return ensure_row_group_fields(out)
        if ag in CANON_CATEGORY_FOR_GROUP:
            out["category"] = CANON_CATEGORY_FOR_GROUP[ag]
        return ensure_row_group_fields(out)

    # Try to infer from name/ingredients/tags
    fix = fallback_group_from_text(name, ingr, cats)
    if fix:
        new_cat, new_ag = fix
        out["alt_group"] = new_ag
        out["category"] = CANON_CATEGORY_FOR_GROUP.get(new_ag, new_cat)
        return ensure_row_group_fields(out)

    # As a final nudge, use name-only high-confidence keywords
    name_l = safe_lower(name)
    for guess_ag, patterns in ALT_KEYWORDS.items():
        if any(re.search(p, name_l) for p in patterns):
            out["alt_group"] = guess_ag
            out["category"] = CANON_CATEGORY_FOR_GROUP.get(guess_ag, out.get("category"))
            return ensure_row_group_fields(out)

    # If still unknown, leave as-is
    return ensure_row_group_fields(out)


def fetch_category_products(category: str, limit: int = 50) -> list[dict[str, Any]]:
    """Search OFF by category term and return normalized products up to `limit`."""
    search_url = f"{OFF_BASE_URL}/cgi/search.pl"
    
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
        r = safe_get(search_url, params=params)
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
                    rule_bases = cast(Sequence[str], rule["bases"])
                    rule_unit = cast(str, rule["unit"])
                    rule_lt = float(cast(int | float, rule["lt"]))
                    disp[field] = display_value(
                        nutriments=n, bases=rule_bases, serving_g=sg,
                        serving_value=features.get(field), unit=rule_unit, lt_threshold=rule_lt
                    )
                features["__display"] = disp
                features["net_carbs_g"] = compute_net_carbs_local(features)
                
                # Ensure we have valid nutritional data
                if features.get("carbs_g") is not None or safe_float(features.get("calories"), 0.0) > 0:
                    products.append(ensure_row_group_fields(features))
                
                # Add small delay to be respectful to the API
                time.sleep(0.1)
                
            except Exception:
                # Skip problematic products
                continue
        
        return products
        
    except Exception as e:
        print(f"Error fetching category products for {category}: {e}")
        return []
