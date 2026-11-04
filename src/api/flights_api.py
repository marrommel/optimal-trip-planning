from datetime import datetime
from html.parser import HTMLParser
import json
import logging
import math
import re
import time
from typing import Any, Optional
import requests
from fast_flights import (
    FlightQuery,
    Passengers,
    create_query,
)
from fast_flights.parser import parse_js
from fast_flights.exceptions import FlightsNotFound
from .api_helpers.progress import ProgressLogger

logger = logging.getLogger(__name__)
_progress = ProgressLogger(logger, "Flights", "flight offers")


class FlightAPIError(RuntimeError):
    """Unusable provider response; never equivalent to no available flights."""


class _FlightPayload(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.chunks = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.active = "ds:1" in (dict(attrs).get("class") or "").split()
            if self.active:
                self.chunks = []

    def handle_endtag(self, tag):
        if tag == "script":
            if self.active:
                self.scripts.append("".join(self.chunks))
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.chunks.append(data)


def _parse_flight_response(html: str):
    """Normalize recommended/other offers for fast-flights' single-section parser."""
    parser = _FlightPayload()
    parser.feed(html)
    if len(parser.scripts) != 1:
        raise ValueError("Expected exactly one flight payload script")
    script = parser.scripts[0]
    marker = re.search(r"""(?:\bdata|["']data["'])\s*:\s*""", script)
    if marker is None:
        raise ValueError("Flight payload script has no data field")
    # Decode the JSON value itself, not the surrounding JavaScript callback.
    # Additional callback metadata can contain any number of commas.
    data = script[marker.end():]
    payload, end = json.JSONDecoder().raw_decode(data)
    trailer = data[end:].lstrip()
    if not trailer.startswith((",", "}")):
        raise ValueError("Unexpected content after flight JSON payload")
    if re.search(r"""(?:^|[,{}])\s*(?:errorHasStatus|["']errorHasStatus["'])\s*:\s*true\b""", trailer):
        raise ValueError("Flight provider reported errorHasStatus; response is not a valid empty search")
    if not isinstance(payload, list) or len(payload) < 8:
        raise ValueError("Unexpected flight payload shape")
    rows = []
    for section in payload[2:4]:
        if section is None:
            continue
        if not isinstance(section, list) or not section:
            raise ValueError("Unexpected flight result section")
        offers = section[0]
        if offers is None:
            continue
        if not isinstance(offers, list):
            raise ValueError("Unexpected flight offer collection")
        rows.extend(offers)
    priced = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError("Unexpected flight offer shape")
        quote = row[1]
        if quote is None or quote == []:
            continue
        if not isinstance(quote, list):
            raise ValueError("Unexpected flight price metadata")
        pair = quote[0]
        if pair is None:
            continue
        if not isinstance(pair, list):
            raise ValueError("Unexpected flight price pair")
        if len(pair) < 2:
            continue
        price = pair[1]
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            continue
        priced.append(row)
    if len(priced) != len(rows):
        logger.warning("Excluded %d unpriced flight offers; retaining %d quoted alternatives",
                       len(rows) - len(priced), len(priced))
    # Google omits empty airline/alliance catalogues, notably on no-service routes.
    # They are display metadata, not evidence that flight results are invalid.
    metadata = payload[7]
    if not isinstance(metadata, list) or len(metadata) < 2:
        raise ValueError("Unexpected flight response metadata")
    catalogues = metadata[1]
    if catalogues is None:
        catalogues = []
    if not isinstance(catalogues, list):
        raise ValueError("Unexpected airline/alliance catalogues")
    normalized = []
    for catalogue in (catalogues + [None, None])[:2]:
        if catalogue is None:
            logger.debug("Provider omitted an optional airline/alliance catalogue")
            catalogue = []
        if not isinstance(catalogue, list) or any(
                not isinstance(entry, list) or len(entry) != 2
                or not all(isinstance(value, str) for value in entry)
                for entry in catalogue):
            raise ValueError("Unexpected airline/alliance catalogue entry")
        normalized.append(catalogue)
    payload[7][1] = normalized
    for row in priced:
        details = row[0]
        if isinstance(details, list) and len(details) >= 3:
            # Missing optional carbon metadata must not abort other priced offers.
            details.extend([None] * max(0, 23 - len(details)))
            extras = details[22]
            if extras is None:
                details[22] = [None] * 9
            elif isinstance(extras, list):
                extras.extend([None] * max(0, 9 - len(extras)))
    payload[3] = [priced]
    return parse_js("data:" + json.dumps(payload) + ",")

# Reusable session with EU GDPR Consent bypass cookies
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
)
_SESSION.cookies.update(
    {
        "SOCS": "CAESHAgBEhJnd3NfMjAyNDA4MTQtMF9SQzIaAmVuIAEaBgiAo_CmBg",
        "CONSENT": "PENDING+987",
    }
)

def _request_flights(params: dict[str, str], route: str, date: str) -> requests.Response:
    for attempt in range(1, 4):
        try:
            response = _SESSION.get(
                "https://www.google.com/travel/flights", params=params, timeout=(10, 30))
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as exc:
            raise FlightAPIError(f"Flight request TLS failure for {route} on {date}") from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 3:
                raise FlightAPIError(
                    f"Flight request failed for {route} on {date} after 3 attempts; "
                    "no empty result was substituted") from exc
            delay = 2 ** attempt
            logger.warning("Flight request for %s on %s failed (%s), attempt %d/3; retrying in %ds",
                           route, date, type(exc).__name__, attempt, delay)
            time.sleep(delay)
    raise AssertionError("Flight request retry loop ended without a response or error")


def _to_datetime(sdt: Any) -> datetime:
    """Convert fast_flights SimpleDatetime(date=(Y, M, D), time=(H, M)) to Python datetime."""
    year, month, day = sdt.date
    hour, minute = sdt.time
    return datetime(year, month, day, hour, minute)


def calculate_total_duration_minutes(flight_item: Any) -> tuple[int, int, int]:
    """Calculate airport-to-airport time in minutes (in-air time + layover time).

    Layover timestamps are local to the same hub. Without provider offsets this
    preliminary filter cannot resolve a daylight-saving transition at that hub.
    The optimizer separately calculates elapsed duration from zoned endpoints.

    Returns:
        tuple[int, int, int]: (total_duration_mins, total_air_mins, total_layover_mins)
    """
    legs = flight_item.flights
    air_mins = sum(leg.duration for leg in legs)

    layover_mins = 0
    for i in range(len(legs) - 1):
        arr_at_hub = _to_datetime(legs[i].arrival)
        dep_from_hub = _to_datetime(legs[i + 1].departure)
        diff = int((dep_from_hub - arr_at_hub).total_seconds() // 60)
        if diff < 0:
            raise FlightAPIError("Flight legs have overlapping local layover timestamps")
        layover_mins += diff

    return air_mins + layover_mins, air_mins, layover_mins


def search_flights(
    date: str,
    from_airport: str,
    to_airport: str,
    max_price: Optional[int] = 1000,
    max_stops: int = 1,
    max_duration_minutes: Optional[int] = 720,
    time_window: Optional[tuple[int, int]] = None,
    carry_on_bags: int = 0,
    checked_bags: int = 0,
    adults: int = 1,
    currency: str = "EUR",
) -> list[dict[str, Any]]:
    """Search Google Flights and filter by stops, true total duration (incl. layover), and time window.

    Args:
        date (str): Flight date in "YYYY-MM-DD" format.
        from_airport (str): 3-letter IATA origin airport code (e.g., "STR").
        to_airport (str): 3-letter IATA destination airport code (e.g., "AUH").
        max_price (int, optional): Maximum ticket price in `currency`. Defaults to 1000.
        max_stops (int, optional): Maximum allowed stops (0 = direct only, 1 = max 1 stop). Defaults to 1.
        max_duration_minutes (int, optional): Max airport-to-airport travel time
            (flight + layover) for both direct and connecting flights. Defaults to 720 (12h).
        time_window (tuple[int, int], optional): Allowed departure hour window (start_hour, end_hour),
            e.g. (6, 18) for 06:00 to 18:00. Defaults to None (any time).
        carry_on_bags (int, optional): Number of carry-on bags. Defaults to 0.
        checked_bags (int, optional): Number of checked bags. Defaults to 0.
        adults (int, optional): Number of adult passengers. Defaults to 1.
        currency (str, optional): Currency code. Defaults to "EUR".

    Returns:
        list[dict[str, Any]]: Filtered flights sorted by (price, stops, total_duration_mins).
            co2_kg is the provider's complete itinerary emission in kilograms,
            converted from grams without an additional aviation multiplier.
            Offers without emissions are excluded with a warning.
    """
    earliest_dep = time_window[0] if time_window else None
    latest_dep = time_window[1] if time_window else None

    flight_query = FlightQuery(
        date=date,
        from_airport=from_airport.upper(),
        to_airport=to_airport.upper(),
        #max_stops=max_stops,
        #earliest_departure_hour=earliest_dep,
        #latest_departure_hour=latest_dep,
        #max_duration_minutes=max_duration_minutes,
    )

    query = create_query(
        flights=[flight_query],
        seat="economy",
        trip="one-way",
        currency=currency,
        max_stops=max_stops,
        max_price=max_price,
        carry_on_bags=carry_on_bags,
        checked_bags=checked_bags,
        passengers=Passengers(adults=adults),
    )

    response = _request_flights(
        {
            "tfs": query.to_str(),
            "hl": "en",
            "gl": "de",
            "curr": currency,
            "tfu": "EgQIABABIgA",
        },
        f"{from_airport}->{to_airport}", date,
    )

    # Missing payload also occurs for blocked/changed responses: fail closed.
    if "ds:1" not in response.text:
        raise FlightAPIError("Google Flights response has no recognizable flight payload")

    try:
        raw_flights = _parse_flight_response(response.text)
    except (AttributeError, TypeError, IndexError, ValueError, KeyError, FlightsNotFound) as exc:
        raise FlightAPIError(f"Cannot parse flights for {from_airport}->{to_airport} on {date}") from exc
    _progress.record(len(raw_flights))
    if not raw_flights:
        logger.info("No priced flight offers for %s->%s on %s", from_airport, to_airport, date)

    filtered_flights: list[dict[str, Any]] = []
    missing_emissions = 0

    for item in raw_flights:
        if not item.flights or item.price is None:
            continue

        # 1. Price check
        if max_price is not None and item.price > max_price:
            continue

        stops = len(item.flights) - 1
        if stops > max_stops:
            continue

        # 2. Optional Departure Time Window check
        dep_dt = _to_datetime(item.flights[0].departure)
        arr_dt = _to_datetime(item.flights[-1].arrival)
        if time_window is not None:
            start_hr, end_hr = time_window
            dep_decimal = dep_dt.hour + (dep_dt.minute / 60.0)
            if not (start_hr <= dep_decimal <= end_hr):
                continue

        # 3. True Total Duration Check (in-air + layovers)
        total_mins, air_mins, layover_mins = calculate_total_duration_minutes(item)

        if max_duration_minutes is not None and total_mins > max_duration_minutes:
            continue

        emission = getattr(getattr(item, "carbon", None), "emission", None)
        if emission is None:
            missing_emissions += 1
            continue
        if (isinstance(emission, bool) or not isinstance(emission, (int, float))
                or not math.isfinite(emission) or emission <= 0):
            raise FlightAPIError(f"Invalid provider emissions for {from_airport}->{to_airport} on {date}")

        filtered_flights.append(
            {
                "price": item.price,
                "currency": currency,
                "airlines": item.airlines,
                "stops": stops,
                "from_airport": item.flights[0].from_airport.code,
                "to_airport": item.flights[-1].to_airport.code,
                "via_airports": [leg.to_airport.code for leg in item.flights[:-1]],
                "departure": dep_dt.strftime("%Y-%m-%d %H:%M"),
                "arrival": arr_dt.strftime("%Y-%m-%d %H:%M"),
                "total_duration_mins": total_mins,
                "total_duration_str": f"{total_mins // 60}h {total_mins % 60:02d}m",
                "air_duration_mins": air_mins,
                "layover_duration_mins": layover_mins,
                "co2_kg": emission / 1000,
                "raw": item,
            }
        )

    if missing_emissions:
        logger.warning("Excluded %d flight offers without provider emissions for %s->%s on %s",
                       missing_emissions, from_airport, to_airport, date)
    # Sort by: 1) Cheapest price, 2) Fewer stops (direct preferred at tie), 3) Shortest total duration
    filtered_flights.sort(
        key=lambda f: (f["price"], f["stops"], f["total_duration_mins"])
    )
    return filtered_flights


def get_cheapest_flight(
    date: str,
    from_airport: str,
    to_airport: str,
    max_price: Optional[int] = 1000,
    max_duration_minutes: Optional[int] = 720,
    time_window: Optional[tuple[int, int]] = None,
    carry_on_bags: int = 0,
    checked_bags: int = 0,
    adults: int = 1,
) -> Optional[dict[str, Any]]:
    """Return the cheapest direct/one-stop flight within the duration limit."""
    flights = search_flights(
        date=date,
        from_airport=from_airport,
        to_airport=to_airport,
        max_price=max_price,
        max_stops=1,
        max_duration_minutes=max_duration_minutes,
        time_window=time_window,
        carry_on_bags=carry_on_bags,
        checked_bags=checked_bags,
        adults=adults,
    )
    return flights[0] if flights else None


if __name__ == "__main__":
    # 1. Fetch all valid flights (Direct OR 1-stop under 12 hours / 720 mins)
    all_valid = search_flights(
        date="2026-12-21",
        from_airport="STR",
        to_airport="AUH",
        max_price=1000,
        max_duration_minutes=720,  # 12 hours max including layover
        time_window=None,          # Optional: e.g. (6, 18) for 06:00 - 18:00 departure
    )

    print(f"Found {len(all_valid)} matching flights:")
    for f in all_valid:
        via = (
            f" (via {','.join(f['via_airports'])}, layover {f['layover_duration_mins']}m)"
            if f["stops"]
            else " (Direct)"
        )
        print(
            f"  €{f['price']} | {f['airlines']} | "
            f"{f['departure']} -> {f['arrival']} | "
            f"Total: {f['total_duration_str']}{via}"
        )

    # 2. Or directly grab the single cheapest matching flight
    best = get_cheapest_flight(
        date="2026-12-21",
        from_airport="STR",
        to_airport="AUH",
        max_duration_minutes=720,
    )
    print("\nCheapest Valid Flight:")
    print(best)
