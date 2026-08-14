from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESERVATION_LOCK = threading.RLock()
VALID_RESERVATION_STATUSES = {"confirmed", "cancelled"}
PRICE_TIER_THRESHOLD_YUAN = 100


class DataStoreError(RuntimeError):
    """A controlled error raised for missing or malformed CSV state."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def data_path(filename: str, override: Optional[Path] = None) -> Path:
    return Path(override) if override is not None else DATA_DIR / filename


def read_csv_rows(path: Path, required_fields: Iterable[str]) -> List[Dict[str, str]]:
    required: Set[str] = set(required_fields)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = set(reader.fieldnames or [])
            missing = required - headers
            if missing:
                names = ", ".join(sorted(missing))
                raise DataStoreError(
                    "invalid_csv",
                    f"{path.name} is missing required columns: {names}.",
                )
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise DataStoreError(
            "missing_csv", f"Required data file {path.name} was not found."
        ) from exc
    except OSError as exc:
        raise DataStoreError(
            "csv_read_failed", f"Could not read {path.name}: {exc}.", retryable=True
        ) from exc
    except (csv.Error, UnicodeError) as exc:
        raise DataStoreError(
            "invalid_csv", f"Could not read {path.name}: {exc}."
        ) from exc


def read_reservations(path: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(
        path,
        {
            "confirmation_id",
            "session_id",
            "restaurant_id",
            "date",
            "time",
            "party_size",
            "status",
            "created_at",
        },
    )
    for row in rows:
        status = row.get("status", "").strip().lower()
        if status not in VALID_RESERVATION_STATUSES:
            confirmation_id = row.get("confirmation_id", "unknown")
            raise DataStoreError(
                "invalid_reservation_status",
                f"Reservation {confirmation_id} has unsupported status '{status}'.",
            )
    return rows


def error_result(code: str, message: str, retryable: bool = False) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {"code": code, "message": message, "retryable": retryable},
    }


def storage_error_result(error: DataStoreError) -> Dict[str, Any]:
    return error_result(error.code, error.message, error.retryable)
