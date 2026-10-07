from datetime import UTC, datetime

from ontario_nowcast.data.s3 import S3Object
from ontario_nowcast.data.sample import _select_cadence


def _object(stamp: str) -> S3Object:
    return S3Object(
        key=f"CONUS/PrecipRate_00.00/20240101/MRMS_PrecipRate_00.00_{stamp}.grib2.gz",
        size=1,
        etag="x",
    )


def test_cadence_preserves_gap_without_phase_shift() -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 1, 0, 18, tzinfo=UTC)
    objects = [
        _object("20240101-000000"),
        _object("20240101-000600"),
        _object("20240101-001400"),
        _object("20240101-001800"),
    ]
    selected = _select_cadence(objects, start, end, 6)
    assert [item is None for _, item in selected] == [False, False, True, False]
