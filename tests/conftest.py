from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "restaurants.csv").write_text(
        "restaurant_id,name,cuisine,average_price,rating,hours\n"
        "R001,Jade Pavilion,Chinese,40,4.2,11:00-21:30\n"
        "R002,Imperial Garden,Chinese,180,4.8,11:00-22:00\n"
        "R003,Seoul Kitchen,Korean,50,4.3,11:00-21:30\n"
        "R004,Hanok Table,Korean,140,4.7,17:00-22:00\n"
        "R005,Lion City Cafe,Singaporean,35,4.1,10:30-21:00\n"
        "R006,Orchid House,Singaporean,130,4.6,11:30-22:00\n",
        encoding="utf-8",
    )
    (tmp_path / "availability.csv").write_text(
        "restaurant_id,weekday,time,total_seats\n"
        "R001,Everyday,18:30,8\n"
        "R001,Everyday,19:00,4\n"
        "R001,Everyday,19:30,8\n"
        "R002,Everyday,19:00,8\n"
        "R003,Everyday,19:00,8\n"
        "R004,Everyday,18:30,8\n"
        "R004,Everyday,19:00,4\n"
        "R004,Everyday,19:30,8\n"
        "R005,Everyday,19:00,8\n"
        "R006,Everyday,19:00,8\n",
        encoding="utf-8",
    )
    (tmp_path / "reservations.csv").write_text(
        "confirmation_id,session_id,restaurant_id,date,time,party_size,status,created_at\n",
        encoding="utf-8",
    )
    return tmp_path
