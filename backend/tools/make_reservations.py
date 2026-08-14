from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from backend.tools.check_availability import check_availability
from backend.tools.common import (
    DataStoreError,
    RESERVATION_LOCK,
    data_path,
    read_reservations,
    storage_error_result,
)


RESERVATION_FIELDS = [
    "confirmation_id",
    "session_id",
    "restaurant_id",
    "date",
    "time",
    "party_size",
    "status",
    "created_at",
]


def _next_confirmation_id(rows: List[Dict[str, str]]) -> str:
    highest = 0
    for row in rows:
        raw_id = row.get("confirmation_id", "")
        if raw_id.startswith("RSV") and raw_id[3:].isdigit():
            highest = max(highest, int(raw_id[3:]))
    return f"RSV{highest + 1:04d}"


def make_reservation(
    restaurant_id: str,
    date: str,
    time: str,
    party_size: int,
    session_id: str,
    restaurants_path: Optional[Path] = None,
    availability_path: Optional[Path] = None,
    reservations_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Atomically recheck capacity and append an idempotent reservation row."""
    reservation_file = data_path("reservations.csv", reservations_path)
    with RESERVATION_LOCK:
        try:
            rows = read_reservations(reservation_file)
            existing = next(
                (
                    row
                    for row in rows
                    if row["status"].strip().lower() == "confirmed"
                    and row["session_id"] == session_id
                    and row["restaurant_id"] == restaurant_id
                    and row["date"] == date
                    and row["time"] == time
                    and row["party_size"] == str(party_size)
                ),
                None,
            )
            availability = check_availability(
                restaurant_id,
                date,
                time,
                party_size,
                restaurants_path=restaurants_path,
                availability_path=availability_path,
                reservations_path=reservation_file,
            )
            if not availability.get("ok"):
                return availability
            if existing is not None:
                return {
                    "ok": True,
                    "duplicate": True,
                    "reservation": {
                        **existing,
                        "restaurant_name": availability["restaurant_name"],
                        "party_size": int(existing["party_size"]),
                    },
                }
            if not availability.get("available"):
                return {
                    "ok": False,
                    "error": {
                        "code": "slot_unavailable",
                        "message": "The requested slot no longer has enough seats.",
                        "retryable": True,
                    },
                    "restaurant_id": restaurant_id,
                    "restaurant_name": availability["restaurant_name"],
                    "date": date,
                    "time": time,
                    "party_size": party_size,
                    "alternatives": availability.get("alternatives", []),
                }

            confirmation_id = _next_confirmation_id(rows)
            row = {
                "confirmation_id": confirmation_id,
                "session_id": session_id,
                "restaurant_id": restaurant_id,
                "date": date,
                "time": time,
                "party_size": str(party_size),
                "status": "confirmed",
                "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
                    timespec="seconds"
                ),
            }
            temp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    newline="",
                    dir=reservation_file.parent,
                    prefix=f".{reservation_file.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=RESERVATION_FIELDS,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows([*rows, row])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, reservation_file)
                temp_path = None
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

            return {
                "ok": True,
                "duplicate": False,
                "reservation": {
                    **row,
                    "restaurant_name": availability["restaurant_name"],
                    "party_size": party_size,
                },
            }
        except (DataStoreError, OSError) as exc:
            if isinstance(exc, DataStoreError):
                return storage_error_result(exc)
            return {
                "ok": False,
                "error": {
                    "code": "reservation_write_failed",
                    "message": f"Could not save the reservation: {exc}.",
                    "retryable": True,
                },
            }
