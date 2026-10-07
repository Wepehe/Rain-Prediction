from datetime import UTC, datetime

from ontario_nowcast.data.multimodal import _goes_scan_time


def test_goes_scan_timestamp_parser() -> None:
    key = "OR_ABI-L2-CMIPC-M6C13_G16_s20241981501179_e20241981503564_c.nc"
    assert _goes_scan_time(key) == datetime(2024, 7, 16, 15, 1, 17, tzinfo=UTC)

