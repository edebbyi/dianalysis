"""Look up a barcode and score the matched item."""

from __future__ import annotations

import traceback

import pandas as pd

from .pipeline import score_item
from ..type_defs import ModelLike


def score_by_barcode(barcode: str, model: ModelLike, df_candidates: pd.DataFrame | None = None) -> dict:
    """Fetch a product by barcode, then score it with alternatives."""
    try:
        from ..off_pipeline import OFF_TAGS_MULTI, fetch_and_normalize_off, fetch_category_products, infer_alt_group_for_item

        item = fetch_and_normalize_off(barcode)
        item = infer_alt_group_for_item(item)

        pool = None
        ag = str(item.get("alt_group", "") or "").lower()
        alternatives_source = "database"
        alternatives_count = 0

        if df_candidates is not None and not df_candidates.empty and ag and "alt_group" in df_candidates.columns:
            pool = df_candidates[df_candidates["alt_group"].str.lower() == ag]
            if not pool.empty:
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
                        alternatives_source = "dynamic"
                        alternatives_count = len(pool)
                        print(f"Created pool with {alternatives_count} alternatives from OpenFoodFacts")

        result = score_item(item, model, pool if pool is not None and not pool.empty else df_candidates)
        result["alternatives_source"] = alternatives_source
        result["alternatives_count"] = alternatives_count
        return result

    except ImportError:
        return {"barcode": barcode, "error": "off_pipeline module not available"}
    except Exception as e:
        traceback.print_exc()
        return {"barcode": barcode, "error": f"lookup failed: {e}"}
