"""Calendar and elapsed-time arithmetic shared by decoding and verification.

Circadian constraint model. A destination night counts only if it provides
MIN_REST_HOURS of continuous rest inside the local sleep window, and a stay
counts only if it yields MIN_USABLE_HOURS_PER_CITY of local daylight. Human
chronobiology, not calendar arithmetic, defines feasibility.
"""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import config
from models import Connection, Visit


def city_zone(city: str) -> ZoneInfo:
    return ZoneInfo(config.CITY_TIMEZONES[city])


def local_time(day: date, hour: float, city: str) -> datetime:
    return datetime.combine(day, time(), city_zone(city)) + timedelta(hours=hour)


def localize_schedule(value: str, city: str) -> datetime:
    """Reject ambiguous/missing local flight times instead of guessing a DST fold."""
    parsed = datetime.fromisoformat(value)
    zone = city_zone(city)
    if parsed.tzinfo is not None:
        return parsed.astimezone(zone)
    first, second = parsed.replace(tzinfo=zone, fold=0), parsed.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        raise ValueError(f"Ambiguous or nonexistent flight time in {city}: {value}; provider offset required")
    return first


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Travel timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def elapsed_hours(start: datetime, end: datetime) -> float:
    return (utc(end) - utc(start)).total_seconds() / 3600


def shift_hours(value: datetime, hours: float) -> datetime:
    return (utc(value) + timedelta(hours=hours)).astimezone(value.tzinfo)


def departure_days() -> range:
    first, last = map(date.fromisoformat, config.TRAVEL_RANGE)
    count = (last - first).days + 2 - config.TOTAL_DAYS
    if count <= 0:
        raise ValueError("Travel window is too short for TOTAL_DAYS")
    return range(count)


def trip_bounds(start_day: int = 0) -> tuple[datetime, datetime]:
    if type(start_day) is not int or start_day not in departure_days():
        raise ValueError("Departure date is outside the allowed trip-start window")
    start = date.fromisoformat(config.TRAVEL_RANGE[0]) + timedelta(days=start_day)
    end = start + timedelta(days=config.TOTAL_DAYS)
    return local_time(start, 0, config.START_CITY), local_time(end, 0, config.START_CITY)


def travel_window(conn: Connection) -> tuple[datetime, datetime]:
    if conn.departure_time is None or conn.arrival_time is None:
        raise ValueError(f"Missing timestamps for {conn.src_name} -> {conn.dest_name}")
    return (shift_hours(conn.departure_time, -conn.access_hours),
            shift_hours(conn.arrival_time, conn.egress_hours))


def journey_eligible(conn: Connection) -> bool:
    """Limit the complete scheduled connection, excluding local access and egress."""
    if conn.departure_time is None or conn.arrival_time is None:
        raise ValueError(f"Missing timestamps for {conn.src_name} -> {conn.dest_name}")
    hours = elapsed_hours(conn.departure_time, conn.arrival_time)
    if hours <= 0:
        raise ValueError(f"Non-positive journey duration for {conn.src_name} -> {conn.dest_name}")
    return hours <= config.MAX_TRANSFER_HOURS


def overlap_hours(start: datetime, end: datetime, left: datetime, right: datetime) -> float:
    return max(0.0, (min(utc(end), utc(right)) - max(utc(start), utc(left))).total_seconds() / 3600)


ClockWindows = dict[tuple[str, date, float, float, bool], tuple[datetime, datetime]]


def sleep_overlap(start: datetime, end: datetime, city: str,
                  calendar: ClockWindows | None = None) -> float:
    return _clock_overlap(start, end, city, config.SLEEP_START_HOUR,
                          config.SLEEP_END_HOUR, True, calendar)


def clock_window_overlap(start: datetime, end: datetime, city: str,
                         from_hour: int, to_hour: int, calendar: ClockWindows | None = None) -> float:
    """Hours of [start, end] inside the daily local from_hour-to_hour window of city."""
    return _clock_overlap(start, end, city, from_hour, to_hour, False, calendar)


def _clock_overlap(start: datetime, end: datetime, city: str, from_hour: float, to_hour: float,
                   overnight: bool, calendar: ClockWindows | None) -> float:
    day = start.astimezone(city_zone(city)).date() - timedelta(days=1)
    last = end.astimezone(city_zone(city)).date()
    start_utc, end_utc = utc(start), utc(end)
    hours = 0.0
    while day <= last:
        key = (city, day, from_hour, to_hour, overnight)
        window = calendar.get(key) if calendar is not None else None
        if window is None:
            window = (utc(local_time(day, from_hour, city)),
                      utc(local_time(day + timedelta(days=int(overnight)), to_hour, city)))
            if calendar is not None:
                calendar[key] = window
        hours += max(0.0, (min(end_utc, window[1]) - max(start_utc, window[0])).total_seconds() / 3600)
        day += timedelta(days=1)
    return hours


def minimum_nights(city: str) -> int:
    return config.MIN_NIGHTS_BY_CITY.get(city, config.MIN_DAYS_PER_CITY)


def make_visit(city_id: int, city: str, arrival: datetime, departure: datetime,
               calendar: dict[tuple[str, date], tuple[datetime, ...]] | None = None) -> Visit:
    """Circadian model: a night counts only with MIN_REST_HOURS of rest in the local sleep window,
    a stay only with MIN_USABLE_HOURS_PER_CITY of local daylight."""
    arrival = arrival.astimezone(city_zone(city))
    departure = departure.astimezone(city_zone(city))
    first, last = arrival.date(), departure.date()
    nights = (last - first).days
    rest_nights, usable = 0, 0.0
    arrival_utc, departure_utc = utc(arrival), utc(departure)
    rest_start = arrival_utc
    day = first
    while day <= last:
        key = (city, day)
        windows = calendar.get(key) if calendar is not None else None
        if windows is None:
            windows = (utc(local_time(day, config.DAYTIME_START_HOUR, city)),
                       utc(local_time(day, config.DAYTIME_END_HOUR, city)),
                       utc(local_time(day, config.SLEEP_START_HOUR, city)),
                       utc(local_time(day + timedelta(days=1), config.SLEEP_END_HOUR, city)),
                       utc(local_time(day, config.CHECK_IN_HOUR, city)))
            if calendar is not None:
                calendar[key] = windows
        if day == first:
            # Early arrival does not grant access to a room booked for the previous night.
            rest_start = max(arrival_utc, windows[4])
        usable += max(0.0, (min(departure_utc, windows[1]) - max(arrival_utc, windows[0])).total_seconds() / 3600)
        rest = max(0.0, (min(departure_utc, windows[3]) - max(rest_start, windows[2])).total_seconds() / 3600)
        if rest >= config.MIN_REST_HOURS:
            rest_nights += 1
        day += timedelta(days=1)
    return Visit(city_id, arrival, departure, nights, rest_nights, usable)
