from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from backend.tools.common import (
    DataStoreError,
    PRICE_TIER_THRESHOLD_YUAN,
    data_path,
    error_result,
    read_csv_rows,
    storage_error_result,
)


SUPPORTED_CUISINES = {"chinese", "korean", "singaporean"}
SUPPORTED_TIERS = {"high", "low"}


def search_restaurants(
    cuisine: str,
    tier: str,
    restaurants_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a cuisine match from the requested fixed yuan price tier."""
    normalized_cuisine = cuisine.strip().lower()
    normalized_tier = tier.strip().lower()

    if normalized_cuisine not in SUPPORTED_CUISINES:
        return error_result(
            "unsupported_cuisine",
            "Cuisine must be Chinese, Korean, or Singaporean.",
        )
    if normalized_tier not in SUPPORTED_TIERS:
        return error_result("unsupported_tier", "Tier must be high or low.")

    path = data_path("restaurants.csv", restaurants_path)
    try:
        rows = read_csv_rows(
            path,
            {"restaurant_id", "name", "cuisine", "average_price", "rating", "hours"},
        )
        matches = [
            row
            for row in rows
            if row.get("cuisine", "").strip().lower() == normalized_cuisine
        ]
        if not matches:
            return error_result(
                "no_restaurant",
                f"No {normalized_cuisine.title()} restaurant is available in the demo data.",
            )

        for row in matches:
            try:
                row["average_price"] = int(row["average_price"])
                row["rating"] = float(row["rating"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataStoreError(
                    "invalid_csv",
                    f"Restaurant {row.get('restaurant_id', 'unknown')} has invalid numeric data.",
                ) from exc

        eligible = [
            row
            for row in matches
            if (
                row["average_price"] > PRICE_TIER_THRESHOLD_YUAN
                if normalized_tier == "high"
                else row["average_price"] <= PRICE_TIER_THRESHOLD_YUAN
            )
        ]
        if not eligible:
            comparison = "above" if normalized_tier == "high" else "at or below"
            return error_result(
                "no_restaurant",
                f"No {normalized_cuisine.title()} restaurant has an average price "
                f"{comparison} ¥{PRICE_TIER_THRESHOLD_YUAN} for the {normalized_tier} tier.",
            )

        chooser = max if normalized_tier == "high" else min
        selected = chooser(eligible, key=lambda row: row["average_price"])
        return {
            "ok": True,
            "restaurant": selected,
            "tier": normalized_tier,
            "tier_threshold_yuan": PRICE_TIER_THRESHOLD_YUAN,
        }
    except DataStoreError as exc:
        return storage_error_result(exc)


def restaurant_recommender_skill(
    cuisine: str,
    tier: str,
    data_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible wrapper for the original teammate tool name."""
    result = search_restaurants(
        cuisine,
        tier,
        restaurants_path=Path(data_path) if data_path else None,
    )
    return result.get("restaurant") if result.get("ok") else None
