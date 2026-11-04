import csv
from hashlib import sha256
import logging
import json
import math
import re
import time
from pathlib import Path
import pandas as pd
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from pydantic import AwareDatetime, BaseModel, Field, ValidationError, field_validator, model_validator

import config
from models import Connection, Hotel
from config import START_CITY, DESTINATIONS, HOME_AIRPORTS, TRAVEL_RANGE

from api.flights_api import search_flights
from api.hotels_api import HotelAPIError, search_hotels
from api.train_api import ConnectionResult, EuropeTransitScraper, TransitAPIError
from timing import city_zone, elapsed_hours, localize_schedule

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
TRANSIT_COLUMNS = [
    "day", "date", "src_idx", "dest_idx", "src_city", "dest_city", "mode",
    "price", "dep_time", "duration", "provider", "co2", "station_src", "station_dest",
    "source_api", "journey_id", "quoted_at", "arrival_time", "departure_time", "stops",
    "is_price_estimated", "is_co2_estimated",
]


class TransitCache(BaseModel):
    schema_version: Literal[3] = 3
    emissions_source: Literal["transit-api"] = "transit-api"
    cities: list[str]
    travel_range: tuple[str, str]
    fetched_at: AwareDatetime
    connections: list[ConnectionResult]


def _transit_frame(quotes: list[ConnectionResult]) -> pd.DataFrame:
    start_date = datetime.strptime(TRAVEL_RANGE[0], "%Y-%m-%d").date()
    city_indices = {city: index for index, city in CITIES.items()}
    records = []
    for quote in quotes:
        departure = quote.departure_time.astimezone(city_zone(quote.origin_city))
        day = departure.date()
        records.append({
            "day": (day - start_date).days, "date": day.isoformat(),
            "src_idx": city_indices[quote.origin_city], "dest_idx": city_indices[quote.destination_city],
            "src_city": quote.origin_city, "dest_city": quote.destination_city,
            "mode": quote.mode, "price": quote.price_eur,
            "dep_time": departure.strftime("%H:%M"),
            "duration": quote.duration_minutes / 60,
            "provider": " / ".join(quote.operators), "co2": quote.co2_kg,
            "station_src": quote.station_src, "station_dest": quote.station_dest,
            "source_api": quote.source_api, "journey_id": quote.journey_id,
            "quoted_at": quote.quoted_at.isoformat(), "arrival_time": quote.arrival_time.isoformat(),
            "departure_time": quote.departure_time.isoformat(), "stops": quote.stops,
            "is_price_estimated": quote.is_price_estimated,
            "is_co2_estimated": quote.is_co2_estimated,
        })
    return pd.DataFrame(records, columns=TRANSIT_COLUMNS)

# --- MAPPING DATA ---
CITIES_LIST = [START_CITY] + DESTINATIONS
CITIES = {i: city for i, city in enumerate(CITIES_LIST)}

# Extended mapping to include the cities for all HOME_AIRPORTS
CITY_AIRPORTS = {
    "Stuttgart": ["STR", "FKB"],
    "Frankfurt": ["FRA", "HHN"],
    "Munich": ["MUC"],
    "Cologne": ["CGN"],
    "London": ["LHR", "LGW", "STN", "LTN"],
    "Paris": ["CDG", "ORY", "BVA"],
    "Madrid": ["MAD"],
    "Barcelona": ["BCN"],
    "Rome": ["FCO", "CIA"]
}


def get_city_for_airport(airport_code: str) -> str:
    """Reverse lookup: Find which city an airport belongs to."""
    for city, airports in CITY_AIRPORTS.items():
        if airport_code in airports:
            return city
    return "Unknown"


# In-memory cache so we don't query Stuttgart->Frankfurt train 10 times for the same day
_TRANSIT_API_CACHE = {}


def get_airport_transit(airport_code: str, date_str: str, scraper: EuropeTransitScraper):
    """Dynamically fetches train/bus data from START_CITY to the airport's base city."""
    target_city = get_city_for_airport(airport_code)

    # This legacy city lookup is not an airport transfer quote.
    if target_city == START_CITY:
        raise TransitAPIError("Local airport access is not priced by the intercity transit API")

    if target_city == "Unknown":
        raise TransitAPIError(f"No city mapping for airport {airport_code}")

    # 3. Check memory cache to save API calls
    cache_key = (date_str, target_city)
    if cache_key in _TRANSIT_API_CACHE:
        return _TRANSIT_API_CACHE[cache_key]

    # 4. Query the actual Transit API
    print(f"    🚆 Fetching transit: {START_CITY} -> {target_city} for {airport_code} on {date_str}...")
    connections = scraper.get_cheapest_in_timespan(
        origin=START_CITY, destination=target_city, date_str=date_str,
        start_time="00:00", end_time="23:59",
    )
    if not connections:
        raise TransitAPIError(f"No priced transit from {START_CITY} to {target_city} on {date_str}")
    best = connections[0]
    cost, duration_hours, provider = best.price_eur, best.duration_minutes / 60, " / ".join(best.operators)

    _TRANSIT_API_CACHE[cache_key] = (cost, duration_hours, provider)
    return cost, duration_hours, provider


# ---------------------------------------------------------
# 1. FLIGHTS CACHING
# ---------------------------------------------------------
class FlightQuote(BaseModel):
    src_city: str
    dest_city: str
    src_airport: str
    dest_airport: str
    departure_time: AwareDatetime
    arrival_time: AwareDatetime
    quoted_at: AwareDatetime
    price: float = Field(gt=0, allow_inf_nan=False)
    air_minutes: int = Field(gt=0)
    co2_kg: float = Field(gt=0, allow_inf_nan=False, strict=True)
    stops: int = Field(ge=0)
    airline: str

    @model_validator(mode="after")
    def validate_route(self):
        if self.src_city not in CITIES_LIST or self.dest_city not in CITIES_LIST or self.src_city == self.dest_city:
            raise ValueError("Flight contains an unknown or repeated city")
        for city, airport in ((self.src_city, self.src_airport), (self.dest_city, self.dest_airport)):
            allowed = HOME_AIRPORTS if city == START_CITY else CITY_AIRPORTS.get(city, [])
            if airport not in allowed:
                raise ValueError("Flight airport does not belong to the configured endpoint")
        hours = elapsed_hours(self.departure_time, self.arrival_time)
        if hours <= 0 or self.air_minutes > hours * 60 + 1:
            raise ValueError("Flight timestamps disagree with air duration")
        return self


class FlightCache(BaseModel):
    schema_version: Literal[3] = 3
    emissions_source: Literal["google-flights"] = "google-flights"
    cities: list[str]
    home_airports: list[str]
    travel_range: tuple[str, str]
    fetched_at: AwareDatetime
    adults: Literal[1] = 1
    carry_on_bags: Literal[0] = 0
    checked_bags: Literal[0] = 0
    currency: Literal["EUR"] = "EUR"
    query_max_duration_minutes: int | None = Field(default=None, gt=0)
    quotes: list[FlightQuote]


def _flight_frame(quotes: list[FlightQuote]) -> pd.DataFrame:
    start_date = datetime.fromisoformat(TRAVEL_RANGE[0]).date()
    indices = {name: index for index, name in CITIES.items()}
    records = []
    for quote in quotes:
        departure = quote.departure_time.astimezone(city_zone(quote.src_city))
        record = quote.model_dump(mode="json")
        record.update(
            day=(departure.date() - start_date).days,
            src_idx=indices[quote.src_city], dest_idx=indices[quote.dest_city],
            dep_time=departure.strftime("%H:%M"),
            duration=elapsed_hours(quote.departure_time, quote.arrival_time),
            co2=quote.co2_kg,
        )
        records.append(record)
    return pd.DataFrame(records)


def fetch_and_cache_flights():
    cache_file = CACHE_DIR / "flight_quotes.json"
    now = datetime.now(timezone.utc)
    if cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("schema_version") != 3:
                raise ValueError("Flight cache requires schema 3 with API emissions; refreshing without estimates")
            cache = FlightCache.model_validate(saved)
            valid = (
                    cache.cities == CITIES_LIST
                    and cache.home_airports == HOME_AIRPORTS
                    and cache.travel_range == tuple(TRAVEL_RANGE)
                    and all(
                        TRAVEL_RANGE[0]
                        <= q.departure_time.astimezone(city_zone(q.src_city)).date().isoformat()
                        <= TRAVEL_RANGE[1]
                        for q in cache.quotes
                    )
            )
            if valid:
                print(f"Loaded flight quotes ({cache_file})")
                return _flight_frame(cache.quotes)

            logger.warning("Ignoring differently configured flight cache.")
        except (ValidationError, ValueError) as exc:
            logger.warning("Ignoring invalid flight cache: %s", exc)

    print("Fetching flight options from API...")
    quotes = []
    start = datetime.fromisoformat(TRAVEL_RANGE[0])
    end = datetime.fromisoformat(TRAVEL_RANGE[1])
    for day in range((end - start).days + 1):
        date_str = (start + timedelta(days=day)).date().isoformat()
        for src in CITIES_LIST:
            for dest in CITIES_LIST:
                if src == dest:
                    continue
                for src_airport in (HOME_AIRPORTS if src == START_CITY else CITY_AIRPORTS.get(src, [])):
                    for dest_airport in (HOME_AIRPORTS if dest == START_CITY else CITY_AIRPORTS.get(dest, [])):
                        time.sleep(1.0)
                        flights = search_flights(
                            date_str, src_airport, dest_airport, max_price=None,
                            max_duration_minutes=math.ceil(config.MAX_TRANSFER_HOURS * 60))
                        quoted_at = datetime.now(timezone.utc)
                        for flight in flights:
                            if flight["from_airport"] != src_airport or flight["to_airport"] != dest_airport:
                                raise ValueError("Flight provider returned different airports")
                            if flight.get("co2_kg") is None:
                                logger.warning("Excluded flight without provider emissions: %s->%s on %s",
                                               src_airport, dest_airport, date_str)
                                continue
                            quotes.append(FlightQuote(
                                src_city=src, dest_city=dest, src_airport=src_airport, dest_airport=dest_airport,
                                departure_time=localize_schedule(flight["departure"], src),
                                arrival_time=localize_schedule(flight["arrival"], dest),
                                quoted_at=quoted_at, price=flight["price"],
                                air_minutes=flight["air_duration_mins"], stops=flight["stops"],
                                co2_kg=flight["co2_kg"],
                                airline=" / ".join(flight["airlines"]),
                            ))
    cache = FlightCache(cities=CITIES_LIST, home_airports=HOME_AIRPORTS, travel_range=tuple(TRAVEL_RANGE),
                        fetched_at=now, quotes=quotes,
                        query_max_duration_minutes=math.ceil(config.MAX_TRANSFER_HOURS * 60))
    temp = cache_file.with_suffix(".tmp")
    temp.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
    temp.replace(cache_file)
    return _flight_frame(quotes)


# ---------------------------------------------------------
# 2. TRANSIT CACHING (TRAINS & BUSES BETWEEN DESTINATIONS)
# ---------------------------------------------------------
def fetch_and_cache_transit():
    """Cache both transit sources, including their returned per-person CO2 values.

    The API currently estimates rail fares and transit emissions. Preserve those
    values and flags; neither loading nor optimization estimates emissions again.
    """
    cache_file = CACHE_DIR / "transit_quotes.json"
    now = datetime.now(timezone.utc)
    if cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("schema_version") != 3:
                raise ValueError("Transit cache requires schema 3 with DB/Flix coverage and API emissions")
            cache = TransitCache.model_validate(saved)
            valid = (
                    cache.cities == CITIES_LIST
                    and cache.travel_range == tuple(TRAVEL_RANGE)
                    and all(
                        quote.origin_city in CITIES_LIST
                        and quote.destination_city in CITIES_LIST
                        and TRAVEL_RANGE[0] <= quote.departure_time.date().isoformat() <= TRAVEL_RANGE[1]
                        and "05:00" <= quote.departure_time.strftime("%H:%M") <= "23:59"
                        for quote in cache.connections
                    )
            )
            if valid:
                print(f"Loaded cached DB/Flix transit options ({cache_file})")
                return _transit_frame(cache.connections)
            logger.warning("Ignoring differently configured transit cache")
        except (ValidationError, ValueError) as exc:
            logger.warning("Ignoring invalid transit quote cache: %s", exc)
    if (CACHE_DIR / "transit_cache.csv").exists():
        logger.warning("Legacy transit_cache.csv is not used: its routes and fares are unverified")

    print("Fetching train and bus departures...")
    quotes = []
    scraper = EuropeTransitScraper()

    start_date = datetime.strptime(TRAVEL_RANGE[0], "%Y-%m-%d")
    end_date = datetime.strptime(TRAVEL_RANGE[1], "%Y-%m-%d")
    total_dates = (end_date - start_date).days

    for day in range(total_dates + 1):
        current_date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")

        for src_idx, src_city in CITIES.items():
            for dest_idx, dest_city in CITIES.items():
                if src_idx == dest_idx: continue

                quotes.extend(scraper.get_cheapest_in_timespan(
                    origin=src_city, destination=dest_city, date_str=current_date,
                    start_time="05:00", end_time="23:59",
                ))

    cache = TransitCache(cities=CITIES_LIST, travel_range=tuple(TRAVEL_RANGE),
                         fetched_at=now, connections=quotes)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(".tmp")
    temp_file.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
    temp_file.replace(cache_file)
    print(f"✅ Saved {len(quotes)} transit options with API CO2 to {cache_file}")
    return _transit_frame(quotes)


# ---------------------------------------------------------
# 3. HOTEL CACHING
# ---------------------------------------------------------
class HotelSearchSettings(BaseModel):
    min_stars: int = Field(ge=1, le=5)
    min_rating: float = Field(ge=0, le=5, allow_inf_nan=False)
    min_reviews: int = Field(ge=0)
    max_price_per_night: float | None = Field(gt=0, allow_inf_nan=False)
    include_unstarred: bool
    adults: int = Field(gt=0, strict=True)
    currency: Literal["EUR"] = "EUR"


def parse_clock_hour(text: str) -> float | None:
    """Parse '2:00 PM' / '14:00' into local hours since midnight."""
    match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*", text.replace("\u202f", " "))
    if not match:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), (match[3] or "").lower()
    if minute > 59 or hour > 24 or (meridiem and not 1 <= hour <= 12):
        return None
    if meridiem:
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    return hour + minute / 60 if hour * 60 + minute <= 24 * 60 else None


class HotelQuote(BaseModel):
    city: str = Field(min_length=1)
    name: str = Field(min_length=1)
    property_token: str = Field(min_length=1)
    hex_id: str = ""
    category: str = Field(min_length=1)
    check_in_date: date
    check_out_date: date
    price_per_night: float = Field(gt=0, allow_inf_nan=False)
    base_stay_price: float = Field(ge=0, allow_inf_nan=False)
    taxes_and_fees: float = Field(ge=0, allow_inf_nan=False)
    total_price: float = Field(gt=0, allow_inf_nan=False)
    currency: Literal["EUR"]
    stars: int = Field(ge=0, le=5)
    rating: float = Field(ge=0, le=5, allow_inf_nan=False)
    reviews_count: int = Field(ge=0)
    co2_kg_per_night: float = Field(ge=0, allow_inf_nan=False)
    total_co2_kg: float = Field(ge=0, allow_inf_nan=False)
    check_in_time: str | None = None   # Earliest check-in as quoted by the provider, e.g. "2:00 PM"
    check_out_time: str | None = None  # Latest check-out as quoted by the provider
    quoted_at: AwareDatetime

    @field_validator("check_in_time", "check_out_time", mode="before")
    @classmethod
    def blank_time_is_missing(cls, value):
        if value is None or not str(value).strip():
            return None
        if parse_clock_hour(str(value)) is None:
            raise ValueError(f"Unrecognized hotel clock time: {value!r}")
        return str(value).strip()

    @model_validator(mode="after")
    def validate_night(self):
        if (self.check_out_date - self.check_in_date).days != 1:
            raise ValueError("Hotel cache requires individual one-night searches")
        if not self.name.strip() or not self.property_token.strip():
            raise ValueError("Hotel identity must not be blank")
        if not math.isclose(self.base_stay_price + self.taxes_and_fees, self.total_price,
                            rel_tol=0, abs_tol=0.03):
            raise ValueError("Hotel total disagrees with its base price and taxes/fees")
        return self


class HotelCache(BaseModel):
    schema_version: Literal[4] = 4
    cities: list[str]
    travel_range: tuple[str, str]
    settings: HotelSearchSettings
    fetched_at: AwareDatetime
    csv_sha256: str


def fetch_and_cache_hotels() -> pd.DataFrame:
    """Cache a complete city/night search grid, including searches with no matches.

    The CSV holds raw offers; its JSON manifest records settings, timestamps and checksum.
    Interrupted/failed refreshes never make a partial grid look complete.
    """
    settings = HotelSearchSettings(
        min_stars=config.HOTEL_MIN_STARS, min_rating=config.HOTEL_MIN_GUEST_RATING,
        min_reviews=config.HOTEL_MIN_REVIEWS_COUNT, max_price_per_night=config.HOTEL_MAX_PRICE_PER_NIGHT,
        include_unstarred=config.HOTEL_INCLUDE_UNSTARRED, adults=config.NUM_TRAVELLERS,
    )
    delay = config.HOTEL_REQUEST_DELAY_SECONDS
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("Hotel request delay must be finite and nonnegative")
    start, end = (date.fromisoformat(value) for value in TRAVEL_RANGE)
    if start >= end:
        raise ValueError("Hotel travel range must allow at least one night")
    cities = [city for index, city in CITIES.items() if index != 0]
    cache_file = CACHE_DIR / "hotel_quotes.csv"
    manifest_file = cache_file.with_suffix(".json")

    def validate_quotes(quotes: list[HotelQuote]):
        for quote in quotes:
            if (quote.city not in cities or not start <= quote.check_in_date < end
                    or quote.check_out_date > end
                    or (quote.stars < settings.min_stars
                        and not (quote.stars == 0 and (settings.include_unstarred or settings.min_stars == 1)))
                    or quote.rating < settings.min_rating
                    or quote.reviews_count < settings.min_reviews
                    or (settings.max_price_per_night is not None
                        and quote.total_price > settings.max_price_per_night)):
                raise ValueError(f"Hotel quote does not match search settings: {quote.name}")

    if cache_file.exists() and manifest_file.exists():
        try:
            cache = HotelCache.model_validate_json(manifest_file.read_text(encoding="utf-8"))
            if (cache.cities != cities or cache.travel_range != tuple(TRAVEL_RANGE)
                    or cache.settings != settings
                    or cache.csv_sha256 != sha256(cache_file.read_bytes()).hexdigest()):
                raise ValueError("Hotel cache is incomplete or differently configured")
            with cache_file.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(HotelQuote.model_fields):
                    raise ValueError("Unexpected hotel CSV columns")
                quotes = [HotelQuote.model_validate(row) for row in reader]
            validate_quotes(quotes)
            print(f"Loaded hotel quotes from cache ({cache_file})", flush=True)
            return pd.DataFrame([q.model_dump(mode="json") for q in quotes], columns=HotelQuote.model_fields)
        except (ValueError, csv.Error) as exc:
            logger.warning("Ignoring invalid hotel cache: %s", exc)
    elif cache_file.exists() or manifest_file.exists():
        logger.warning("Ignoring incomplete hotel cache: CSV and metadata are both required")

    print("Fetching hotel quotes from API...", flush=True)
    quotes = []
    request_count = 0
    for city in cities:
        for day in range((end - start).days):
            check_in = start + timedelta(days=day)
            check_out = check_in + timedelta(days=1)
            if request_count:
                time.sleep(delay)
            request_count += 1
            offers = search_hotels(
                city=city, check_in_date=check_in.isoformat(), check_out_date=check_out.isoformat(),
                **settings.model_dump(),
            )
            quoted_at = datetime.now(timezone.utc)
            for offer in offers:
                try:
                    quote = HotelQuote.model_validate({
                        **offer, "hex_id": offer.get("hex_id") or "",
                        "property_token": offer.get("property_token") or offer["name"],
                        "category": offer.get("category") or "gcid:hotel", "quoted_at": quoted_at,
                    })
                    if (quote.city, quote.check_in_date, quote.check_out_date) != (city, check_in, check_out):
                        raise ValueError("Hotel provider returned a different city or stay")
                    validate_quotes([quote])
                    quotes.append(quote)
                except ValueError as exc:
                    raise HotelAPIError(f"Invalid hotel offer for {city} on {check_in}") from exc
    now = datetime.now(timezone.utc)
    temp_csv = cache_file.with_suffix(".csv.tmp")
    temp_manifest = manifest_file.with_suffix(".json.tmp")
    try:
        with temp_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(HotelQuote.model_fields))
            writer.writeheader()
            writer.writerows(q.model_dump(mode="json") for q in quotes)
        cache = HotelCache(
            cities=cities, travel_range=tuple(TRAVEL_RANGE), settings=settings,
            fetched_at=now, csv_sha256=sha256(temp_csv.read_bytes()).hexdigest(),
        )
        temp_manifest.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
        temp_csv.replace(cache_file)
        temp_manifest.replace(manifest_file)
    finally:
        temp_csv.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
    return pd.DataFrame([q.model_dump(mode="json") for q in quotes], columns=HotelQuote.model_fields)


# ---------------------------------------------------------
# 4. BUILD OPTIMIZER DATABASE
# ---------------------------------------------------------
def load_connections() -> list[Connection]:
    connections = []

    # Load Flights
    flights_df = fetch_and_cache_flights()
    for _, row in flights_df.iterrows():
        hh, mm = map(int, str(row['dep_time']).split(":"))
        dep_hour = hh + (mm / 60.0)

        connections.append(Connection(
            day=int(row['day']), src=int(row['src_idx']), dest=int(row['dest_idx']), mode='flight',
            dep_hour=dep_hour, duration=row['duration'], cost=row['price'] * config.NUM_TRAVELLERS,
                        co2=row['co2'] * config.NUM_TRAVELLERS,
                        src_name=row['src_city'], dest_name=row['dest_city'],
                        provider=row['airline'], station_src=row['src_airport'], station_dest=row['dest_airport'],
            fare_source="google-flights", quoted_at=row["quoted_at"],
            departure_time=datetime.fromisoformat(row["departure_time"]),
            arrival_time=datetime.fromisoformat(row["arrival_time"]), stops=int(row["stops"]),
            access_hours=config.AIRPORT_ACCESS_HOURS[row["src_airport"]] + config.FLIGHT_CHECK_IN_HOURS,
            egress_hours=config.AIRPORT_ACCESS_HOURS[row["dest_airport"]] + config.FLIGHT_EXIT_HOURS,
            co2_method="Google Flights itinerary emissions.",
        ))

    # Load Trains & Buses
    transit_df = fetch_and_cache_transit()
    for _, row in transit_df.iterrows():
        hh, mm = map(int, str(row['dep_time']).split(":"))
        dep_hour = hh + (mm / 60.0)

        connections.append(Connection(
            day=int(row['day']), src=int(row['src_idx']), dest=int(row['dest_idx']), mode=row['mode'],
            dep_hour=dep_hour, duration=row['duration'], cost=row['price'] * config.NUM_TRAVELLERS,
                        co2=row['co2'] * config.NUM_TRAVELLERS,
                        src_name=row['src_city'], dest_name=row['dest_city'],
                        provider=row['provider'], station_src=row['station_src'], station_dest=row['station_dest'],
            fare_source=row['source_api'], quoted_at=row['quoted_at'],
            arrival_time=datetime.fromisoformat(row['arrival_time']),
            departure_time=datetime.fromisoformat(row["departure_time"]), stops=int(row["stops"]),
            journey_id=row["journey_id"], access_hours=config.STATION_ACCESS_HOURS,
            egress_hours=config.STATION_EXIT_HOURS,
            co2_method=f"Transit API {row['source_api']} co2_kg (cached"
                       + (" estimate)." if row["is_co2_estimated"] else ")."),
        ))

    return connections


def load_hotels() -> list[Hotel]:
    """Build per-hotel nightly estimates; an absent day means no usable quote, not sold out."""
    if not config.OPTIMIZE_HOTELS:
        return []
    frame = fetch_and_cache_hotels()

    indices = {city: index for index, city in CITIES.items()}
    start = date.fromisoformat(TRAVEL_RANGE[0])
    hotels = []
    frame["property_id"] = frame["hex_id"].where(frame["hex_id"] != "", frame["property_token"])
    for (city, token), offers in frame.groupby(["city", "property_id"], sort=True):
        name = offers.iloc[0]["name"]
        stars = int(offers["stars"].min())
        categories = set(offers["category"])
        if len(categories) != 1:
            raise ValueError(f"Conflicting property categories for {name}")
        is_hostel = next(iter(categories)) == "gcid:hostel"
        # CO2 is the provider-API estimate cached with each quote; never recomputed here.
        co2 = float(offers["co2_kg_per_night"].max())
        discomfort = config.HOTEL_DISCOMFORT_PER_NIGHT_BY_STARS[stars]
        if is_hostel:
            discomfort = config.HOTEL_HOSTEL_DISCOMFORT_PER_NIGHT
        # Conservative: latest quoted earliest-check-in and earliest quoted latest-check-out.
        check_ins = [h for h in map(parse_clock_hour, offers["check_in_time"].dropna().astype(str)) if h is not None]
        check_outs = [h for h in map(parse_clock_hour, offers["check_out_time"].dropna().astype(str)) if h is not None]
        if any(not math.isfinite(value) or value < 0 for value in (co2, discomfort)):
            raise ValueError("Hotel carbon and discomfort estimates must be finite and nonnegative")
        rates = {
            (date.fromisoformat(night) - start).days: float(price)
            for night, price in offers.groupby("check_in_date")["total_price"].min().items()
        }
        hotels.append(Hotel(
            city=indices[city], tier=1 if is_hostel or stars in (1, 2) else 2 if stars in (0, 3) else 3,
            daily_rates=rates, discomfort=discomfort, co2_per_night=round(co2, 2),
            city_name=city, hotel_name=name, property_token=token,
            check_in_hour=max(check_ins) if check_ins else None,
            check_out_hour=min(check_outs) if check_outs else None,
        ))
    return hotels


CONNECTIONS_DB: list[Connection] = []
CONNECTIONS_BY_ROUTE: dict[tuple[int, int, int], list[Connection]] = {}
HOTELS_DB: list[Hotel] = []


def initialize_database():
    """Explicit startup avoids network calls when importing optimizer modules."""
    CONNECTIONS_DB[:] = load_connections()
    CONNECTIONS_BY_ROUTE.clear()
    for connection in CONNECTIONS_DB:
        validate_connection(connection)
        key = (connection.day, connection.src, connection.dest)
        CONNECTIONS_BY_ROUTE.setdefault(key, []).append(connection)
    HOTELS_DB[:] = load_hotels()


def validate_connection(conn: Connection):
    if conn.departure_time is None or conn.arrival_time is None:
        raise ValueError("Every connection requires departure and arrival timestamps")
    values = (conn.cost, conn.co2, conn.duration, conn.access_hours, conn.egress_hours)
    if (any(not math.isfinite(v) or v < 0 for v in values) or conn.duration <= 0 or conn.stops < 0
            or conn.mode not in config.TRAVEL_DISCOMFORT_PER_HOUR
            or conn.src not in CITIES or conn.dest not in CITIES or conn.src == conn.dest):
        raise ValueError("Invalid connection endpoints, mode or numeric values")
    if not math.isclose(elapsed_hours(conn.departure_time, conn.arrival_time), conn.duration, abs_tol=1 / 60):
        raise ValueError("Connection duration disagrees with timestamps")
    local_departure = conn.departure_time.astimezone(city_zone(CITIES[conn.src]))
    day = (local_departure.date() - datetime.fromisoformat(TRAVEL_RANGE[0]).date()).days
    if day != conn.day or not math.isclose(conn.dep_hour, local_departure.hour + local_departure.minute / 60, abs_tol=1 / 60):
        raise ValueError("Departure day/time disagrees with local timestamp")


def print_database_summary():
    """Prints the entire search space data in tabular format for inspection."""
    print("=" * 125)
    print(f"{'🌍 TRIP PLANNER - FULL DATABASE DUMP 🌍':^125}")
    print("=" * 125)

    print("🏨 HOTELS (All Loaded Data)")
    print(f"{'City':<12} | {'Hotel Name':<28} | {'Tier':<5} | {'Cost Range (€)':<16} | {'Unquoted Days':<20}")
    print("-" * 125)

    for h in sorted(HOTELS_DB, key=lambda x: (x.city, x.tier)):
        tier = {1: "Budget", 2: "Mid", 3: "Prem"}[h.tier]
        rates = list(h.daily_rates.values())
        price_str = f"€{min(rates):.0f} - €{max(rates):.0f}" if rates else "NO VACANCY"
        quoted_nights = (date.fromisoformat(TRAVEL_RANGE[1]) - date.fromisoformat(TRAVEL_RANGE[0])).days
        missing_days = ",".join(str(d) for d in range(quoted_nights) if d not in h.daily_rates)
        print(f"{h.city_name:<12} | {h.hotel_name:<28} | {tier:<5} | {price_str:<16} | {missing_days or 'None':<20}")

    for mode, title in [('flight', '✈️ FLIGHTS'), ('train', '🚆 TRAINS'),
                        ('bus', '🚌 BUSES'), ('train+bus', '🚆🚌 MIXED TRANSIT')]:
        print(f"\n{title}")
        print(
            f"{'Day':<4} | {'Provider':<15} | {'Route':<25} | {'Stations (Src -> Dest)':<35} | {'Dep':<5} | {'Dur(h)':<6} | {'Cost':<7} | {'CO2':<5}")
        print("-" * 125)

        mode_conns = sorted([c for c in CONNECTIONS_DB if c.mode == mode],
                            key=lambda x: (x.day, x.src_name, x.dest_name, x.dep_hour))
        for c in mode_conns[:100]:  # Print first 100 to avoid console freeze
            route = f"{c.src_name} -> {c.dest_name}"
            stations = f"{c.station_src} -> {c.station_dest}"
            hour, minute = divmod(round(c.dep_hour * 60), 60)
            dep_str = f"{hour:02d}:{minute:02d}"
            print(
                f"{c.day:<4} | {c.provider[:14]:<15} | {route[:24]:<25} | {stations[:34]:<35} | {dep_str:<5} | {c.duration:<6.1f} | €{c.cost:<6.0f} | {c.co2:<5.0f}")
        if len(mode_conns) > 100: print(f"... and {len(mode_conns) - 100} more {mode}s")
    print("=" * 125 + "\n")