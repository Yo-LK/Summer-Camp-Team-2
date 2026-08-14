from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from backend.tools.check_availability import check_availability
from backend.tools.make_reservations import make_reservation
from backend.tools.search_restaurants import search_restaurants


FUTURE_DATE = "2099-08-15"


def paths(data_dir: Path):
    return {
        "restaurants_path": data_dir / "restaurants.csv",
        "availability_path": data_dir / "availability.csv",
        "reservations_path": data_dir / "reservations.csv",
    }


def test_search_selects_lowest_and_highest_tiers(data_dir: Path):
    low = search_restaurants("CHINESE", "low", data_dir / "restaurants.csv")
    high = search_restaurants("Chinese", "HIGH", data_dir / "restaurants.csv")

    assert low["ok"] is True
    assert low["restaurant"]["restaurant_id"] == "R001"
    assert high["restaurant"]["restaurant_id"] == "R002"


def test_search_rejects_unsupported_preferences(data_dir: Path):
    cuisine = search_restaurants("Italian", "low", data_dir / "restaurants.csv")
    tier = search_restaurants("Chinese", "medium", data_dir / "restaurants.csv")

    assert cuisine["error"]["code"] == "unsupported_cuisine"
    assert tier["error"]["code"] == "unsupported_tier"


def test_search_enforces_fixed_yuan_tier_threshold(data_dir: Path):
    restaurant_file = data_dir / "restaurants.csv"
    restaurant_file.write_text(
        "restaurant_id,name,cuisine,average_price,rating,hours\n"
        "R100,Exactly One Hundred,Chinese,100,4.5,11:00-22:00\n"
        "R101,Above One Hundred,Chinese,101,4.4,11:00-22:00\n"
        "R102,Only Low,Korean,80,4.3,11:00-22:00\n"
        "R103,Only High,Singaporean,120,4.6,11:00-22:00\n",
        encoding="utf-8",
    )

    low = search_restaurants("Chinese", "low", restaurant_file)
    high = search_restaurants("Chinese", "high", restaurant_file)
    missing_high = search_restaurants("Korean", "high", restaurant_file)
    missing_low = search_restaurants("Singaporean", "low", restaurant_file)

    assert low["restaurant"]["restaurant_id"] == "R100"
    assert low["restaurant"]["average_price"] == 100
    assert high["restaurant"]["restaurant_id"] == "R101"
    assert high["restaurant"]["average_price"] == 101
    assert missing_high["error"]["code"] == "no_restaurant"
    assert missing_low["error"]["code"] == "no_restaurant"


def test_availability_validates_and_offers_nearby_slots(data_dir: Path):
    kwargs = paths(data_dir)
    past = check_availability(
        "R001", "2020-01-01", "19:00", 2, today=date(2026, 1, 1), **kwargs
    )
    invalid_party = check_availability(
        "R001", FUTURE_DATE, "19:00", 13, **kwargs
    )

    assert past["error"]["code"] == "past_date"
    assert invalid_party["error"]["code"] == "invalid_party_size"

    booked = make_reservation("R001", FUTURE_DATE, "19:00", 4, "first", **kwargs)
    unavailable = check_availability("R001", FUTURE_DATE, "19:00", 2, **kwargs)

    assert booked["ok"] is True
    assert unavailable["ok"] is True
    assert unavailable["available"] is False
    assert [slot["time"] for slot in unavailable["alternatives"]] == ["18:30", "19:30"]


def test_reservation_is_persistent_and_idempotent(data_dir: Path):
    kwargs = paths(data_dir)
    first = make_reservation("R003", FUTURE_DATE, "19:00", 3, "session-a", **kwargs)
    repeated = make_reservation("R003", FUTURE_DATE, "19:00", 3, "session-a", **kwargs)
    availability = check_availability("R003", FUTURE_DATE, "19:00", 1, **kwargs)

    assert first["reservation"]["confirmation_id"] == "RSV0001"
    assert repeated["duplicate"] is True
    assert repeated["reservation"]["confirmation_id"] == "RSV0001"
    assert availability["remaining_seats"] == 5

    with (data_dir / "reservations.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_cancelled_row_releases_capacity(data_dir: Path):
    kwargs = paths(data_dir)
    made = make_reservation("R001", FUTURE_DATE, "19:00", 4, "session-a", **kwargs)
    assert made["ok"] is True
    assert check_availability("R001", FUTURE_DATE, "19:00", 1, **kwargs)["available"] is False

    reservation_file = data_dir / "reservations.csv"
    with reservation_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    rows[0]["status"] = "cancelled"
    with reservation_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    available = check_availability("R001", FUTURE_DATE, "19:00", 4, **kwargs)
    assert available["available"] is True
    assert available["remaining_seats"] == 4


def test_unknown_reservation_status_is_a_controlled_error(data_dir: Path):
    (data_dir / "reservations.csv").write_text(
        "confirmation_id,session_id,restaurant_id,date,time,party_size,status,created_at\n"
        "RSV0001,s1,R001,2099-08-15,19:00,2,pending,2099-01-01T00:00:00\n",
        encoding="utf-8",
    )

    result = check_availability("R001", FUTURE_DATE, "19:00", 2, **paths(data_dir))
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_reservation_status"


def test_missing_storage_file_is_a_controlled_error(data_dir: Path):
    (data_dir / "reservations.csv").unlink()

    result = check_availability("R001", FUTURE_DATE, "19:00", 2, **paths(data_dir))

    assert result["ok"] is False
    assert result["error"]["code"] == "missing_csv"
