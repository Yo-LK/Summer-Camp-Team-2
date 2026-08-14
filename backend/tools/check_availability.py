from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from backend.tools.common import (
    DataStoreError,
    data_path,
    error_result,
    read_csv_rows,
    read_reservations,
    storage_error_result,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _parse_date(
    raw_date: str, today: Optional[date_type]
) -> Tuple[Optional[date_type], Optional[Dict[str, Any]]]:
    try:
        requested_date = date_type.fromisoformat(raw_date)
    except ValueError:
        return None, error_result("invalid_date", "Date must use YYYY-MM-DD format.")
    current_date = today or datetime.now(SHANGHAI).date()
    if requested_date < current_date:
        return None, error_result("past_date", "Reservations cannot be made for a past date.")
    return requested_date, None


def _minutes(raw_time: str) -> int:
    parsed = datetime.strptime(raw_time, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def _booked_seats(
    reservations: List[Dict[str, str]], restaurant_id: str, raw_date: str, raw_time: str
) -> int:
    total = 0
    for row in reservations:
        if (
            row["status"].strip().lower() == "confirmed"
            and row["restaurant_id"] == restaurant_id
            and row["date"] == raw_date
            and row["time"] == raw_time
        ):
            try:
                total += int(row["party_size"])
            except ValueError as exc:
                raise DataStoreError(
                    "invalid_csv",
                    f"Reservation {row.get('confirmation_id', 'unknown')} has an invalid party size.",
                ) from exc
    return total


def check_availability(
    restaurant_id: str,
    date: str,
    time: str,
    party_size: int,
    restaurants_path: Optional[Path] = None,
    availability_path: Optional[Path] = None,
    reservations_path: Optional[Path] = None,
    today: Optional[date_type] = None,
) -> Dict[str, Any]:
    """Check a recurring slot and return up to three nearby alternatives."""
    if not 1 <= party_size <= 12:
        return error_result("invalid_party_size", "Party size must be between 1 and 12.")

    requested_date, date_error = _parse_date(date, today)
    if date_error:
        return date_error
    try:
        requested_minutes = _minutes(time)
    except ValueError:
        return error_result("invalid_time", "Time must use 24-hour HH:MM format.")

    restaurant_file = data_path("restaurants.csv", restaurants_path)
    availability_file = data_path("availability.csv", availability_path)
    reservation_file = data_path("reservations.csv", reservations_path)

    try:
        restaurants = read_csv_rows(
            restaurant_file,
            {"restaurant_id", "name", "cuisine", "average_price", "rating", "hours"},
        )
        restaurant = next(
            (row for row in restaurants if row["restaurant_id"] == restaurant_id), None
        )
        if restaurant is None:
            return error_result(
                "restaurant_not_found", f"Restaurant {restaurant_id} does not exist."
            )

        try:
            opening, closing = restaurant["hours"].split("-", maxsplit=1)
            if not _minutes(opening) <= requested_minutes <= _minutes(closing):
                return error_result(
                    "restaurant_closed",
                    f"{restaurant['name']} is open from {restaurant['hours']}.",
                )
        except (ValueError, KeyError) as exc:
            raise DataStoreError(
                "invalid_csv", f"Restaurant {restaurant_id} has invalid opening hours."
            ) from exc

        rows = read_csv_rows(
            availability_file, {"restaurant_id", "weekday", "time", "total_seats"}
        )
        weekday = requested_date.strftime("%A") if requested_date else ""
        applicable = [
            row
            for row in rows
            if row["restaurant_id"] == restaurant_id
            and row["weekday"].strip().lower() in {weekday.lower(), "everyday"}
        ]
        reservations = read_reservations(reservation_file)

        slot_by_time: Dict[str, int] = {}
        for row in applicable:
            try:
                total_seats = int(row["total_seats"])
                _minutes(row["time"])
                if total_seats < 0:
                    raise ValueError
            except ValueError as exc:
                raise DataStoreError(
                    "invalid_csv",
                    f"Availability for {restaurant_id} contains invalid slot data.",
                ) from exc
            if row["time"] in slot_by_time:
                raise DataStoreError(
                    "invalid_csv",
                    f"Availability for {restaurant_id} contains duplicate slot {row['time']}.",
                )
            slot_by_time[row["time"]] = total_seats

        def remaining(slot_time: str) -> int:
            return max(
                0,
                slot_by_time[slot_time]
                - _booked_seats(reservations, restaurant_id, date, slot_time),
            )

        alternatives = [
            {"time": slot_time, "remaining_seats": remaining(slot_time)}
            for slot_time in slot_by_time
            if slot_time != time and remaining(slot_time) >= party_size
        ]
        alternatives.sort(
            key=lambda item: (
                abs(_minutes(item["time"]) - requested_minutes),
                item["time"],
            )
        )
        alternatives = alternatives[:3]

        if time not in slot_by_time:
            return {
                **error_result(
                    "slot_not_offered",
                    f"{restaurant['name']} does not offer a {time} reservation slot.",
                ),
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant["name"],
                "date": date,
                "time": time,
                "party_size": party_size,
                "alternatives": alternatives,
            }

        seats_left = remaining(time)
        return {
            "ok": True,
            "available": seats_left >= party_size,
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant["name"],
            "date": date,
            "time": time,
            "party_size": party_size,
            "remaining_seats": seats_left,
            "alternatives": [] if seats_left >= party_size else alternatives,
        }
    except DataStoreError as exc:
        return storage_error_result(exc)
