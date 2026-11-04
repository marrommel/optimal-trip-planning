import base64
from datetime import datetime
from html.parser import HTMLParser
import json
import logging
import math
import re
import time
from typing import Any, Optional
import requests

from .api_helpers.progress import ProgressLogger

logger = logging.getLogger(__name__)
_progress = ProgressLogger(logger, "Hotels", "hotel offers")


class HotelAPIError(RuntimeError):
    """Unusable provider response; never equivalent to no available hotels."""


# Cornell Hotel Sustainability Benchmarking (CHSB) / HCMI European averages (kg CO2e / room-night)
_DEFAULT_CO2_KG_BY_STARS: dict[int, float] = {
    0: 15.5,  # Unclassified boutique hotel fallback
    1: 9.0,
    2: 11.5,
    3: 15.5,
    4: 24.0,
    5: 42.0,
}
_HOSTEL_CO2_KG_PER_NIGHT = 7.5


class _HotelPayload(HTMLParser):
    """Extract the `ds:0` AF_initDataCallback script block from Google Travel HTML."""

    def __init__(self):
        super().__init__()
        self.in_script = False
        self.current_chunks: list[str] = []
        self.ds0_script: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        if tag == "script":
            self.in_script = True
            self.current_chunks = []

    def handle_endtag(self, tag: str):
        if tag == "script" and self.in_script:
            self.in_script = False
            joined = "".join(self.current_chunks).strip()
            if "AF_initDataCallback" in joined and "data:" in joined:
                if re.search(r"key:\s*'ds:0'", joined):
                    self.ds0_script = joined

    def handle_data(self, data: str):
        if self.in_script:
            self.current_chunks.append(data)


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

_RAW_QUERY_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}
_LAST_REQUEST_TS: float = 0.0
_MIN_REQUEST_INTERVAL_SEC: float = 0.25


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _protobuf(*fields: tuple[int, int | bytes]) -> bytes:
    return b"".join(
        _varint(number << 3) + _varint(value) if isinstance(value, int)
        else _varint((number << 3) | 2) + _varint(len(value)) + value
        for number, value in fields
    )


def _hotel_search_state(check_in: datetime, check_out: datetime, adults: int, currency: str) -> str:
    """Encode Google's undocumented `ts` search state, not a natural-language date query."""
    def encoded_date(value: datetime) -> bytes:
        return _protobuf((1, value.year), (2, value.month), (3, value.day))

    dates = _protobuf(
        (1, encoded_date(check_in)), (2, encoded_date(check_out)),
        (3, (check_out - check_in).days),
    )
    # ts.3.2.2 contains the date range; ts.3.2.6.1 is the adult count.
    stay = _protobuf((2, dates), (6, _protobuf((1, adults))))
    state = _protobuf(
        (1, 1),
        (2, _protobuf((1, _protobuf((1, 3))), (2, 1))),
        (3, _protobuf((2, stay))),
        (5, _protobuf((1, _protobuf((7, currency.encode("ascii")))))),
    )
    return base64.urlsafe_b64encode(state).decode("ascii").rstrip("=")


def _get(obj: Any, *path: Any) -> Any:
    """Safely traverse nested lists/dicts by positional index or key."""
    cur = obj
    for step in path:
        if cur is None:
            return None
        try:
            cur = cur[step]
        except (IndexError, KeyError, TypeError):
            return None
    return cur


def calculate_stay_nights(check_in_date: str, check_out_date: str) -> int:
    """Calculate the number of hotel nights between check_in_date and check_out_date."""
    dt_in = datetime.strptime(check_in_date, "%Y-%m-%d")
    dt_out = datetime.strptime(check_out_date, "%Y-%m-%d")
    nights = (dt_out - dt_in).days
    if nights <= 0:
        raise ValueError(
            f"check_out_date ({check_out_date}) must be after check_in_date ({check_in_date})"
        )
    return nights


def estimate_hotel_co2_kg(
    stars: int,
    nights: int,
    category: str = "gcid:hotel",
    co2_by_stars: Optional[dict[int, float]] = None,
) -> tuple[float, float]:
    """Calculate nightly and total stay CO2 emissions (kg) using HCMI/CHSB benchmarks."""
    if category == "gcid:hostel":
        nightly_co2 = _HOSTEL_CO2_KG_PER_NIGHT
    else:
        lookup = co2_by_stars or _DEFAULT_CO2_KG_BY_STARS
        clamped_stars = max(0, min(5, int(stars or 0)))
        nightly_co2 = lookup.get(clamped_stars, 15.5)
    nightly_co2 = round(nightly_co2, 2)
    return nightly_co2, round(nightly_co2 * nights, 2)


def _parse_hotel_response(
    html: str,
    expected_check_in: tuple[int, int, int],
    expected_check_out: tuple[int, int, int],
    expected_nights: int,
    expected_adults: int,
    expected_currency: str = "EUR",
) -> list[dict[str, Any]]:
    """Parse Google Hotels' `ds:0` Protobuf array (`397419284`) via exact positional indices."""
    parser = _HotelPayload()
    parser.feed(html)

    if not parser.ds0_script or "data:" not in parser.ds0_script:
        raise HotelAPIError("Google Hotels response has no recognizable ds:0 script payload")

    _, after_data = parser.ds0_script.split("data:", 1)
    payload, _ = json.JSONDecoder().raw_decode(after_data.lstrip())

    # Google echoes the resolved ts state here, even when the result feed is empty.
    context = _get(payload, 1, 0)
    dates = _get(context, 2, 1, 1)
    if dates != [list(expected_check_in), list(expected_check_out), expected_nights]:
        raise HotelAPIError(
            f"Google Hotels search dates are missing or differ: got {dates!r}, "
            f"expected {expected_check_in} -> {expected_check_out} ({expected_nights} nights)"
        )
    adults = _get(context, 2, 1, 5, 0)
    if type(adults) is not int or adults != expected_adults:
        raise HotelAPIError(f"Expected {expected_adults} adults, got {adults!r}")
    currency = _get(context, 4, 0, 6)
    if currency != expected_currency:
        raise HotelAPIError(f"Expected search currency {expected_currency}, got {currency!r}")

    feed = _get(payload, 0, 0, 0, 1)
    if not isinstance(feed, list):
        raise HotelAPIError("Unexpected Google Hotels ds:0 feed shape at payload[0][0][0][1]")

    offers: list[dict[str, Any]] = []
    unpriced_count = 0
    date_rejections = 0

    for item in feed:
        # Widget type 34 is a hotel listing card
        if _get(item, 0) != 34:
            continue

        h = _get(item, 1, "397419284", 0)
        if not isinstance(h, list):
            raise HotelAPIError("Unexpected hotel protobuf container under key '397419284'")

        name = _get(h, 1) or _get(h, 2, 6, 0)
        if not isinstance(name, str) or not name.strip():
            continue

        # 1. Price & Date Verification
        quoted_currency = _get(h, 6, 1, 3) or _get(h, 6, 2, 15)
        if quoted_currency and quoted_currency != expected_currency:
            raise HotelAPIError(f"Expected currency {expected_currency}, got {quoted_currency}")

        conf_in = _get(h, 6, 1, 4, 0)
        conf_out = _get(h, 6, 1, 4, 1)
        conf_nights = _get(h, 6, 1, 4, 2)
        # Google sometimes mixes in cards quoted for other dates; drop them.
        if (conf_in != list(expected_check_in) or conf_out != list(expected_check_out)
                or type(conf_nights) is not int or conf_nights != expected_nights):
            date_rejections += 1
            logger.warning(
                "Skipping %s: quoted %s -> %s (%s nights), expected %s -> %s",
                name, conf_in, conf_out, conf_nights, expected_check_in, expected_check_out,
            )
            continue

        nightly_price = _get(h, 6, 2, 1, 2)
        if nightly_price is None:
            nightly_price = _get(h, 6, 2, 1, 4)
        if (
            isinstance(nightly_price, bool)
            or not isinstance(nightly_price, (int, float))
            or not math.isfinite(nightly_price)
            or nightly_price <= 0
        ):
            unpriced_count += 1
            continue

        tax_breakdown = _get(h, 6, 2, 44)
        if tax_breakdown is not None:
            if not isinstance(tax_breakdown, list) or len(tax_breakdown) < 4:
                raise HotelAPIError(f"Malformed price breakdown for {name}")
            components = [0.0 if value is None else value for value in tax_breakdown[:4]]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in components):
                raise HotelAPIError(f"Invalid price breakdown for {name}")
            base_stay_price = round(float(components[0]), 2)
            taxes_and_fees = round(float(components[1]) + float(components[2]), 2)
            total_price = round(float(components[3]), 2)
        else:
            total_price = round(float(nightly_price) * expected_nights, 2)
            base_stay_price = total_price
            taxes_and_fees = 0.0
        if (any(not math.isfinite(value) or value < 0 for value in (base_stay_price, taxes_and_fees, total_price))
                or total_price <= 0
                or not math.isclose(base_stay_price + taxes_and_fees, total_price, rel_tol=0, abs_tol=0.03)):
            raise HotelAPIError(f"Invalid or inconsistent price breakdown for {name}")

        # 2. Star Class & Property Category (needed internally for CO2 estimate)
        category = _get(h, 2, 31, 0, 0, 0) or "gcid:hotel"
        raw_stars = _get(h, 3, 1)
        stars = int(raw_stars) if isinstance(raw_stars, int) and 1 <= raw_stars <= 5 else 0

        # 3. Guest Rating & Review Count
        rating_pair = _get(h, 7, 0) or _get(h, 7, 8, 0, 1)
        if isinstance(rating_pair, list) and len(rating_pair) >= 2:
            rating = round(float(rating_pair[0] or 0.0), 2)
            reviews_count = int(rating_pair[1] or 0)
        else:
            rating = 0.0
            reviews_count = 0

        # 4. Check-in / Check-out Times (CORRECTED INDEX: 17)
        raw_check_in = _get(h, 2, 17, 0)
        raw_check_out = _get(h, 2, 17, 1)
        check_in_time = raw_check_in.replace("\u202f", " ") if isinstance(raw_check_in, str) else None
        check_out_time = raw_check_out.replace("\u202f", " ") if isinstance(raw_check_out, str) else None

        offers.append(
            {
                "name": name.strip(),
                "category": category,
                "stars": stars,
                "rating": rating,
                "reviews_count": reviews_count,
                "price_per_night": round(float(nightly_price), 2),
                "base_stay_price": base_stay_price,
                "taxes_and_fees": taxes_and_fees,
                "total_price": total_price,
                "check_in_time": check_in_time,
                "check_out_time": check_out_time,
            }
        )

    if unpriced_count > 0:
        logger.warning(
            "Excluded %d unpriced hotel items; retaining %d quoted offers",
            unpriced_count, len(offers),
        )

    if not offers and (date_rejections or unpriced_count):
        raise HotelAPIError(
            "Google Hotels returned no usable quotes: "
            f"{date_rejections} with missing/wrong dates, {unpriced_count} unpriced"
        )

    return offers


def search_hotels(
    city: str,
    check_in_date: str,
    check_out_date: str,
    min_stars: int = 1,
    min_rating: float = 0.0,
    min_reviews: int = 0,
    max_price_per_night: Optional[float] = None,
    max_total_price: Optional[float] = None,
    include_unstarred: bool = False,
    adults: int = 1,
    currency: str = "EUR",
) -> list[dict[str, Any]]:
    """Search Google Hotels for exact dates/occupancy and filter by stars, rating, and price."""
    global _LAST_REQUEST_TS

    nights = calculate_stay_nights(check_in_date, check_out_date)
    if not city.strip() or isinstance(adults, bool) or not isinstance(adults, int) or adults < 1:
        raise ValueError("A city and a positive integer number of adults are required")
    if not 1 <= min_stars <= 5 or not 0 <= min_rating <= 10 or min_reviews < 0:
        raise ValueError("Invalid hotel quality filters")
    for cap in (max_price_per_night, max_total_price):
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("Hotel price caps must be finite and positive")

    city = city.strip()
    currency = currency.strip().upper()
    if re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise ValueError("Currency must be a three-letter code")
    normalized_min_rating = min_rating / 2.0 if min_rating > 5.0 else min_rating
    dt_in = datetime.strptime(check_in_date, "%Y-%m-%d")
    dt_out = datetime.strptime(check_out_date, "%Y-%m-%d")
    expected_in = (dt_in.year, dt_in.month, dt_in.day)
    expected_out = (dt_out.year, dt_out.month, dt_out.day)

    search_state = _hotel_search_state(dt_in, dt_out, adults, currency)
    cache_key = (city.casefold(), search_state)

    if cache_key in _RAW_QUERY_CACHE:
        raw_hotels = _RAW_QUERY_CACHE[cache_key]
    else:
        elapsed = time.monotonic() - _LAST_REQUEST_TS
        if elapsed < _MIN_REQUEST_INTERVAL_SEC:
            time.sleep(_MIN_REQUEST_INTERVAL_SEC - elapsed)

        try:
            response = _SESSION.get(
                "https://www.google.com/travel/hotels",
                params={
                    "q": city,
                    "ts": search_state,
                    "hl": "en",
                    "gl": "de",
                    "curr": currency,
                    "currency": currency,
                },
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HotelAPIError(f"Hotel request failed for {city} ({check_in_date} -> {check_out_date})") from exc
        finally:
            _LAST_REQUEST_TS = time.monotonic()

        if "AF_initDataCallback" not in response.text:
            raise HotelAPIError("Google Hotels response has no recognizable payload")

        try:
            raw_hotels = _parse_hotel_response(
                html=response.text,
                expected_check_in=expected_in,
                expected_check_out=expected_out,
                expected_nights=nights,
                expected_currency=currency,
                expected_adults=adults,
            )
        except (AttributeError, TypeError, IndexError, ValueError, KeyError) as exc:
            raise HotelAPIError(
                f"Cannot parse hotels for {city} ({check_in_date} -> {check_out_date})"
            ) from exc

        _RAW_QUERY_CACHE[cache_key] = raw_hotels
        _progress.record(len(raw_hotels))

    if not raw_hotels:
        logger.info("No priced hotel offers for %s (%s -> %s)", city, check_in_date, check_out_date)

    filtered_hotels: list[dict[str, Any]] = []

    for item in raw_hotels:
        nightly_price = item["price_per_night"]
        total_price = item["total_price"]

        # 1. Price checks
        if max_price_per_night is not None and total_price > max_price_per_night * nights:
            continue
        if max_total_price is not None and total_price > max_total_price:
            continue

        # 2. Star rating check
        if item["stars"] == 0:
            if min_stars > 1 and not include_unstarred:
                continue
        elif item["stars"] < min_stars:
            continue

        # 3. Guest rating & review count checks
        if item["rating"] < normalized_min_rating:
            continue
        if item["reviews_count"] < min_reviews:
            continue

        co2_per_night, total_co2 = estimate_hotel_co2_kg(
            stars=item["stars"],
            nights=nights,
            category=item["category"],
        )

        filtered_hotels.append(
            {
                "name": item["name"],
                "city": city,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "nights": nights,
                "check_in_time": item["check_in_time"],
                "check_out_time": item["check_out_time"],
                "price_per_night": nightly_price,
                "base_stay_price": item["base_stay_price"],
                "taxes_and_fees": item["taxes_and_fees"],
                "total_price": total_price,
                "currency": currency,
                "stars": item["stars"],
                "rating": item["rating"],
                "rating_10_scale": round(item["rating"] * 2.0, 1),
                "reviews_count": item["reviews_count"],
                "co2_kg_per_night": co2_per_night,
                "total_co2_kg": total_co2,
            }
        )

    # Sort by: 1) Cheapest total price, 2) Highest guest rating, 3) Highest stars, 4) Name
    filtered_hotels.sort(
        key=lambda h: (h["total_price"], -h["rating"], -h["stars"], h["name"])
    )
    return filtered_hotels


def get_cheapest_hotel(
    city: str,
    check_in_date: str,
    check_out_date: str,
    min_stars: int = 1,
    min_rating: float = 0.0,
    min_reviews: int = 0,
    max_price_per_night: Optional[float] = None,
    max_total_price: Optional[float] = None,
    include_unstarred: bool = False,
    adults: int = 1,
    currency: str = "EUR",
) -> Optional[dict[str, Any]]:
    """Return the single cheapest hotel in `city` satisfying all minimum quality filters."""
    hotels = search_hotels(
        city=city,
        check_in_date=check_in_date,
        check_out_date=check_out_date,
        min_stars=min_stars,
        min_rating=min_rating,
        min_reviews=min_reviews,
        max_price_per_night=max_price_per_night,
        max_total_price=max_total_price,
        include_unstarred=include_unstarred,
        adults=adults,
        currency=currency,
    )
    return next(iter(hotels), None)


if __name__ == "__main__":
    for test_city in ("Paris", "London", "Rome"):
        matches = search_hotels(
            city=test_city,
            check_in_date="2027-02-02",
            check_out_date="2027-02-04",
            min_stars=3,
            min_rating=3.8,
            max_price_per_night=180.0,
        )
        best = next(iter(matches), None)
        print(
            f"{test_city:10s} ({len(matches):2d} valid) -> \n"
            + (
                f"  Cheapest: €{best['total_price']:6.2f} | {best['stars']}* | {best['rating']}/5 | {best['name']}\n"
                f"  Times: In: {best['check_in_time']} - Out: {best['check_out_time']}\n"
                if best else "  None found\n"
            )
        )