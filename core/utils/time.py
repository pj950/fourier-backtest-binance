from datetime import UTC, datetime


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def now_utc() -> datetime:
    return datetime.now(UTC)


def timestamp_to_datetime(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)


def datetime_to_timestamp(dt: datetime) -> int:
    return int(to_utc(dt).timestamp() * 1000)


def parse_date(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return to_utc(dt)
