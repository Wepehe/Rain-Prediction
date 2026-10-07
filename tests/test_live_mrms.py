from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ontario_nowcast.data.s3 import S3Object
from ontario_nowcast.operational.live_mrms import (
    LiveDataUnavailable,
    download_atomic,
    forecast_cache_identity,
    location_in_domain,
    prepare_live_history,
    regular_history,
    tile_for_location,
)
from ontario_nowcast.operational.live_product import (
    freshness_minutes,
    has_newer_issue,
    location_summary,
    point_from_latlon,
)


def obj(stamp: datetime, size: int = 4) -> S3Object:
    key = f"CONUS/PrecipRate_00.00/{stamp:%Y%m%d}/MRMS_PrecipRate_00.00_{stamp:%Y%m%d-%H%M%S}.grib2.gz"
    return S3Object(key, size, "etag")


def scans(end: datetime, missing: set[int] | None = None):
    missing = missing or set()
    return [obj(end - timedelta(minutes=6 * i)) for i in range(9, -1, -1) if i not in missing]


def test_regular_sequence_and_missing_scan_are_explicit():
    end = datetime(2026, 1, 1, 12, tzinfo=UTC)
    selected = regular_history(scans(end, {4}))
    assert len(selected) == 10
    assert all((b[0] - a[0]) == timedelta(minutes=6) for a, b in pairwise(selected))
    assert sum(item is None for _, item in selected) == 1


def test_domain_coordinate_conversion_and_edge_shift():
    assert location_in_domain(43.6532, -79.3832)
    assert not location_in_domain(45.5019, -73.5674)
    centre = tile_for_location(43.6532, -79.3832)
    edge = tile_for_location(41.51, -84.79)
    assert centre.x.shape == edge.x.shape == (128,)
    assert centre.latitude.shape == (128, 128)
    assert 0 <= edge.point_x_index < 128 and 0 <= edge.point_y_index < 128
    with pytest.raises(ValueError, match="outside"):
        tile_for_location(45.5019, -73.5674)


def test_live_history_missing_mask_without_substitution(tmp_path):
    end = datetime(2026, 1, 1, 12, tzinfo=UTC)
    source = scans(end, {4})
    def listing(_): return source
    def downloader(item, root):
        path = root / Path(item.key).name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"data"); return path
    def decoder(path, bbox):
        del path, bbox
        lat = np.linspace(47, 41, 601); lon = np.linspace(-86, -74, 1201)
        return np.ones((601, 1201), np.float32), lat, lon
    live = prepare_live_history(43.6532, -79.3832, cache_dir=tmp_path, now=end,
                                listing=listing, downloader=downloader, decoder=decoder)
    assert live.history_rate.shape == (10, 128, 128)
    assert len(live.missing_timestamps) == 1
    assert not live.validity_mask[5].any()
    assert live.source_keys[5] == ""


def test_too_many_missing_frames_rejected(tmp_path):
    end = datetime(2026, 1, 1, 12, tzinfo=UTC)
    with pytest.raises(LiveDataUnavailable, match="too many"):
        prepare_live_history(43.6, -79.4, cache_dir=tmp_path, now=end,
                             listing=lambda _: scans(end, {2, 4}))


def test_cache_identity_is_stable_and_sensitive():
    tile = tile_for_location(43.6532, -79.3832)
    issue = datetime(2026, 1, 1, tzinfo=UTC)
    first = forecast_cache_identity(tile, issue, "v1", "abc")
    assert first == forecast_cache_identity(tile, issue, "v1", "abc")
    assert first != forecast_cache_identity(tile, issue + timedelta(minutes=6), "v1", "abc")


def test_point_extraction_at_requested_latlon():
    tile = tile_for_location(43.6532, -79.3832)
    metadata = {"grid": {"latitude": tile.latitude, "longitude": tile.longitude}}
    assert point_from_latlon(metadata, 43.6532, -79.3832) == (
        tile.point_y_index,
        tile.point_x_index,
    )


class Response:
    def __init__(self): self.calls = 0
    def raise_for_status(self): pass
    def iter_content(self, _): self.calls += 1; yield b"data"


def test_atomic_download_reuses_immutable_file(tmp_path):
    stamp = datetime(2026, 1, 1, tzinfo=UTC); item = obj(stamp)
    calls = []
    def get(*args, **kwargs): calls.append((args, kwargs)); return Response()
    first = download_atomic(item, tmp_path, get=get)
    second = download_atomic(item, tmp_path, get=get)
    assert first == second and first.read_bytes() == b"data"
    assert len(calls) == 1
    assert not list((tmp_path / "raw").glob("*.part"))


def test_point_summary_refresh_and_threshold():
    from types import SimpleNamespace
    probability = np.zeros((20, 128, 128), np.float32); probability[7:, 3, 4] = 0.35
    rate = np.zeros_like(probability); rate[7:, 3, 4] = 1
    result = SimpleNamespace(residual_probability=probability, residual_rate_mm_h=rate,
                             metadata={"latest_rate_at_location_mm_h": 0.0})
    summary = location_summary(result, 3, 4)
    assert summary["first_wet_lead_minutes"] == 48
    assert "around +48" in summary["status"]
    current = datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert freshness_minutes(current, current + timedelta(minutes=8)) == 8
    assert has_newer_issue(current, [obj(current + timedelta(minutes=6))])


def test_live_modules_have_no_research_manifest_dependency():
    for name in ("live_mrms.py", "live_product.py"):
        source = (Path("src/ontario_nowcast/operational") / name).read_text().lower()
        assert "final_manifest" not in source
        assert "train_dev" not in source
