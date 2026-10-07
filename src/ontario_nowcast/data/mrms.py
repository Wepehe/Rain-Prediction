"""NOAA MRMS precipitation-rate archive discovery."""

from __future__ import annotations

from datetime import UTC, datetime

from .s3 import S3Object, list_public_bucket

BUCKET = "noaa-mrms-pds"
PRODUCT = "PrecipRate_00.00"


def daily_objects(day: datetime) -> list[S3Object]:
    if day.tzinfo is None:
        raise ValueError("day must be timezone-aware")
    utc_day = day.astimezone(UTC)
    prefix = f"CONUS/{PRODUCT}/{utc_day:%Y%m%d}/"
    return list_public_bucket(BUCKET, prefix)


def objects_between(start: datetime, end: datetime) -> list[S3Object]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if end <= start:
        raise ValueError("end must be later than start")
    selected: list[S3Object] = []
    day = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end.astimezone(UTC):
        for obj in daily_objects(day):
            stamp = obj.key.rsplit("_", 1)[-1].split(".grib2", 1)[0]
            observed = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
            if start <= observed <= end:
                selected.append(obj)
        day = day.fromtimestamp(day.timestamp() + 86400, tz=UTC)
    return sorted(selected, key=lambda item: item.key)

