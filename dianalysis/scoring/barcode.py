"""Look up a barcode and score the matched item."""

from __future__ import annotations

import traceback

import pandas as pd

from .pipeline import score_item
from ..type_defs import ModelLike


CORE_FIELDS = ("carbs_g", "sugar_g", "added_sugar_g", "fiber_g", "protein_g", "fat_g", "sodium_mg")
PRIMARY_FIELDS = ("carbs_g", "sugar_g", "added_sugar_g", "fiber_g", "protein_g")


def _is_sufficient_barcode_nutrition(item: dict) -> tuple[bool, str | None]:
    """
    Decide whether barcode nutrition is complete enough for a reliable score.

    Why:
    - Some barcode responses have sparse nutrition labels.
    - We avoid showing a risk score when core fields are mostly missing/zero.
    """
    disp = item.get("__display", {}) or {}
    known_fields = 0
    primary_nonzero = 0
    nonzero_total = 0

    for field in CORE_FIELDS:
        shown = str(disp.get(field, "not listed")).strip().lower()
        if shown and shown != "not listed":
            known_fields += 1

        try:
            val = float(item.get(field)) if item.get(field) is not None else 0.0
        except Exception:
            val = 0.0

        if val > 0.0:
            nonzero_total += 1
            if field in PRIMARY_FIELDS:
                primary_nonzero += 1

    if known_fields < 3:
        return False, "Not enough nutrition fields were listed to compute a reliable score."
    if primary_nonzero == 0:
        return False, "Core nutrition fields were mostly missing or zero for this barcode."
    if nonzero_total <= 1:
        return False, "Only one non-zero nutrient value was available from this barcode."
    return True, None


def score_by_barcode(barcode: str, model: ModelLike, df_candidates: pd.DataFrame | None = None) -> dict:
    """Fetch a product by barcode, then score it with alternatives."""
    try:
        from ..off_pipeline import OFF_TAGS_MULTI, fetch_and_normalize_off, fetch_category_products, infer_alt_group_for_item

        item = fetch_and_normalize_off(barcode)
        item = infer_alt_group_for_item(item)

        pool = None
        ag = str(item.get("alt_group", "") or "").lower()
        fine_group = str(item.get("alt_group_fine", "") or "").lower()
        alternatives_source = "database"
        alternatives_count = 0

        if df_candidates is not None and not df_candidates.empty:
            if fine_group and "alt_group_fine" in df_candidates.columns:
                pool = df_candidates[df_candidates["alt_group_fine"].fillna("").str.lower() == fine_group]
            if (pool is None or pool.empty) and ag and "alt_group" in df_candidates.columns:
                pool = df_candidates[df_candidates["alt_group"].fillna("").str.lower() == ag]
            if pool is not None and not pool.empty:
                alternatives_count = len(pool)

        if pool is None or pool.empty:
            print(f"No alternatives in database for {ag}, fetching from OpenFoodFacts...")
            if ag and ag in OFF_TAGS_MULTI:
                search_tags = OFF_TAGS_MULTI.get(ag, [ag])
                real_alts = []
                for tag in search_tags[:2]:
                    try:
                        print(f"Fetching products for tag: {tag}")
                        products = fetch_category_products(tag, limit=30)
                        print(f"Fetched {len(products)} products for tag {tag}")
                        real_alts.extend(products)
                        if len(real_alts) >= 50:
                            break
                    except Exception as e:
                        print(f"Error fetching {tag}: {e}")
                        continue

                if real_alts:
                    filtered_alts = []
                    query_cat = str(item.get("category", "") or "").lower()
                    for product in real_alts:
                        product_ag = str(product.get("alt_group") or "").lower()
                        product_cat = str(product.get("category") or "").lower()
                        if product_ag == ag or (not product_ag and product_cat == query_cat):
                            filtered_alts.append(product)

                    print(f"Filtered to {len(filtered_alts)} products matching alt_group '{ag}'")
                    if filtered_alts:
                        pool = pd.DataFrame(filtered_alts)
                        if "alt_group" not in pool.columns:
                            pool["alt_group"] = pool["category"]
                        pool["alt_group"] = ag
                        if "category_main" not in pool.columns:
                            pool["category_main"] = pool["category"]
                        if fine_group:
                            pool["alt_group_fine"] = fine_group
                        alternatives_source = "dynamic"
                        alternatives_count = len(pool)
                        print(f"Created pool with {alternatives_count} alternatives from OpenFoodFacts")

        result = score_item(item, model, pool if pool is not None and not pool.empty else df_candidates)

        sufficient, reason = _is_sufficient_barcode_nutrition(item)
        if not sufficient:
            result["insufficient_data"] = True
            result["insufficient_data_reason"] = reason
            result["reasons"] = []
            notes = list(result.get("notes", []) or [])
            notes.append(
                "Score hidden: this barcode did not include enough nutrition detail for a reliable prediction."
            )
            notes.append("Tip: use Manual Entry to input the full nutrition label and get a reliable score.")
            result["notes"] = notes
            result["risk_display"] = "—"

        result["alternatives_source"] = alternatives_source
        result["alternatives_count"] = alternatives_count
        return result

    except ImportError:
        return {"barcode": barcode, "error": "off_pipeline module not available"}
    except ValueError as e:
        return {"barcode": barcode, "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"barcode": barcode, "error": f"lookup failed: {e}"}
