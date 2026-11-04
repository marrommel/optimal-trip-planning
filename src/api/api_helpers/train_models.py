import csv
import logging
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class TransitAPIError(RuntimeError):
    """The provider failed; this is different from a successful empty search."""


# Connected European mainland + UK/Scandinavia rail & bus network.
# Excludes disconnected islands (cy, is, mt) and restricted networks (by, ru, ua).
EUROPE_COUNTRY_CODES = {
    "ad", "al", "at", "ba", "be", "bg", "ch", "cz", "de", "dk",
    "ee", "es", "fi", "fr", "gb", "gi", "gr", "hr", "hu", "ie",
    "it", "li", "lt", "lu", "lv", "mc", "md", "me", "mk", "nl",
    "no", "pl", "pt", "ro", "rs", "se", "si", "sk", "sm", "tr", "va", "xk",
}

# 1,350 km straight-line (~1,690 km rail/road, ~21h max).
# Keeps Paris -> Rome (1,105 km) and Berlin -> Rome (1,180 km), drops Beijing -> Paris or Lisbon -> Helsinki.
MAX_STRAIGHT_DISTANCE_KM = 1350.0
MIN_STRAIGHT_DISTANCE_KM = 5.0
ROUTE_DETOUR_FACTOR = 1.25


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    country: str
    aliases: tuple[str, ...]
    population: int = 0


def _should_replace_city(existing: City, candidate: City, restrict_to_europe: bool = True) -> bool:
    if restrict_to_europe:
        existing_is_eu = existing.country in EUROPE_COUNTRY_CODES
        candidate_is_eu = candidate.country in EUROPE_COUNTRY_CODES
        if candidate_is_eu and not existing_is_eu:
            return True
        if existing_is_eu and not candidate_is_eu:
            return False
    return candidate.population > existing.population


def load_cities_from_geonames(filepath: str, only_europe: bool = True) -> dict[str, City]:
    """Load GeoNames TSV file safely, resolving duplicate city names by population and region."""
    cities_db: dict[str, City] = {}
    path = Path(filepath)

    if not path.is_file():
        logger.error("GeoNames file %r not found. CITIES database will be empty.", str(path))
        return cities_db

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, start=1):
                if not row or row[0].startswith("#") or len(row) < 15:
                    continue
                try:
                    name = row[1].strip()
                    ascii_name = row[2].strip()
                    country_code = row[8].strip().lower()

                    if only_europe and country_code not in EUROPE_COUNTRY_CODES:
                        continue

                    raw_aliases = [a.strip() for a in row[3].split(",") if a.strip()]
                    aliases = tuple(dict.fromkeys([name, ascii_name, *raw_aliases]))
                    lat = float(row[4])
                    lon = float(row[5])
                    population = int(row[14]) if row[14].strip().isdigit() else 0

                    candidate = City(
                        name=name,
                        latitude=lat,
                        longitude=lon,
                        country=country_code,
                        aliases=aliases,
                        population=population,
                    )

                    for key in {name, ascii_name}:
                        if not key:
                            continue
                        existing = cities_db.get(key)
                        if existing is None or _should_replace_city(existing, candidate):
                            cities_db[key] = candidate
                except (ValueError, IndexError) as row_err:
                    logger.debug("Skipping malformed row %d in %s: %s", line_num, filepath, row_err)
    except OSError as exc:
        logger.error("Failed to read GeoNames file %r: %s", str(path), exc)

    return cities_db


def lookup_city(name: str, cities_db: dict[str, City]) -> Optional[City]:
    """Look up a city by exact name, case-insensitive name, or alias (picking highest population)."""
    if not name or not cities_db:
        return None

    clean_name = name.strip()
    if clean_name in cities_db:
        return cities_db[clean_name]

    clean_lower = clean_name.lower()
    for key, city in cities_db.items():
        if key.lower() == clean_lower:
            return city

    best_match: Optional[City] = None
    for city in set(cities_db.values()):
        if any(alias.lower() == clean_lower for alias in city.aliases):
            if best_match is None or _should_replace_city(best_match, city):
                best_match = city
    return best_match


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def normalized_name(name: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", name.casefold())
        if not unicodedata.combining(c)
    ).strip()


def estimate_route_distance_and_co2(origin: City, destination: City, modes: set[str] | list[str]) -> tuple[float, float]:
    """Return (estimated_route_km, estimated_co2_kg) using your 1.25 detour factor and mode rates."""
    straight = distance_km(origin.latitude, origin.longitude, destination.latitude, destination.longitude)
    est_distance = round(straight * ROUTE_DETOUR_FACTOR, 1)
    rate_g_per_km = 27.0 if "bus" in modes else 14.0
    co2_kg = round(est_distance * rate_g_per_km / 1000.0, 2)
    return est_distance, co2_kg


# Load European cities once at module import
CITIES = load_cities_from_geonames(
    str(Path(__file__).resolve().parents[1] / "cities15000.txt"),
    only_europe=True,
)


def resolve_feasible_route(
    origin: str,
    destination: str,
    cities_db: dict[str, City] = CITIES,
    max_straight_km: float = MAX_STRAIGHT_DISTANCE_KM,
) -> Optional[tuple[City, City]]:
    """Fast pre-flight guardrail (<0.1ms).

    Returns (origin_city, dest_city) ONLY if:
    1. Both cities exist in cities15000.txt (and are in Europe due to only_europe=True).
    2. Both are in connected European transit countries (blocks Dubai, Dallas, Tokyo, Cairo, Cyprus, Malta, Iceland).
    3. The route does not cross the Irish Sea (Ireland <-> Continental Europe/UK has no rail/bus tunnel).
    4. Straight-line distance is between 5 km and 1,350 km (allows Paris -> Rome, blocks Beijing -> Paris).
    """
    if not origin or not destination or origin.strip().lower() == destination.strip().lower():
        return None

    orig_city = lookup_city(origin, cities_db)
    dest_city = lookup_city(destination, cities_db)
    if orig_city is None or dest_city is None:
        logger.debug("Guardrail: %r or %r not found in European CITIES database.", origin, destination)
        return None

    if (orig_city.country not in EUROPE_COUNTRY_CODES
            or dest_city.country not in EUROPE_COUNTRY_CODES):
        logger.debug("Guardrail: %s (%s) -> %s (%s) outside Europe.", origin, orig_city.country, destination, dest_city.country)
        return None

    # Ireland ('ie') is only connected to itself (or Northern Ireland 'gb' by island bus, blocked for cross-sea)
    if (orig_city.country == "ie") != (dest_city.country == "ie"):
        logger.debug("Guardrail: %s -> %s crosses the Irish Sea.", origin, destination)
        return None

    straight = distance_km(orig_city.latitude, orig_city.longitude, dest_city.latitude, dest_city.longitude)
    if straight < MIN_STRAIGHT_DISTANCE_KM or straight > max_straight_km:
        logger.debug("Guardrail: %s -> %s straight distance %.1f km outside [%.0f, %.0f] km.",
                     origin, destination, straight, MIN_STRAIGHT_DISTANCE_KM, max_straight_km)
        return None

    return orig_city, dest_city


class ConnectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    journey_id: str = Field(min_length=1)
    source_api: Literal["flixbus", "deutsche_bahn"] = "flixbus"
    origin_city: str
    destination_city: str
    departure_time: AwareDatetime
    arrival_time: AwareDatetime
    duration_minutes: int = Field(gt=0)
    stops: int = Field(ge=0)
    price_eur: float = Field(gt=0, allow_inf_nan=False)
    is_price_estimated: bool = False
    currency: Literal["EUR"] = "EUR"
    quoted_at: AwareDatetime
    operators: list[str] = Field(min_length=1)
    transport_modes: list[Literal["train", "bus"]] = Field(min_length=1)
    station_src: str = Field(min_length=1)
    station_dest: str = Field(min_length=1)
    distance_km: float = Field(gt=0, allow_inf_nan=False)
    co2_kg: float = Field(ge=0, allow_inf_nan=False)
    is_co2_estimated: Literal[True] = True

    @model_validator(mode="after")
    def validate_journey(self) -> Self:
        elapsed = (self.arrival_time - self.departure_time).total_seconds() / 60
        if elapsed <= 0 or abs(elapsed - self.duration_minutes) > 1.5:
            raise ValueError("duration does not match offset-aware timestamps")
        origin = lookup_city(self.origin_city, CITIES)
        destination = lookup_city(self.destination_city, CITIES)
        if origin is None or destination is None:
            raise ValueError("journey contains an unsupported city")
        straight = distance_km(origin.latitude, origin.longitude,
                               destination.latitude, destination.longitude)
        if straight <= 0:
            raise ValueError("origin and destination must differ")
        ceiling = 350 if "train" in self.transport_modes else 130
        if straight / (elapsed / 60) > ceiling:
            raise ValueError("journey exceeds plausible end-to-end speed")
        if abs(self.distance_km - straight * ROUTE_DETOUR_FACTOR) > 0.2:
            raise ValueError("distance does not match requested cities")
        return self

    @property
    def duration_formatted(self) -> str:
        hours, minutes = divmod(self.duration_minutes, 60)
        return f"{hours}h {minutes:02d}m"

    @property
    def mode(self) -> str:
        return "train+bus" if len(set(self.transport_modes)) > 1 else self.transport_modes[0]
