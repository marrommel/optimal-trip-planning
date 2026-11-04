import logging
import time
from datetime import date, datetime, time as dt_time, timezone
from typing import Literal, Self

from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException
from pydantic import AwareDatetime, BaseModel, Field, ValidationError, model_validator

from .train_models import (
    CITIES,
    City,
    ConnectionResult,
    TransitAPIError,
    distance_km,
    estimate_route_distance_and_co2,
    normalized_name,
)

logger = logging.getLogger(__name__)


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(ge=-180, le=180, allow_inf_nan=False)


class FlixCity(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    country: str
    location: Coordinates


class Endpoint(BaseModel):
    date: AwareDatetime
    city_id: str
    station_id: str


class Leg(BaseModel):
    departure: Endpoint
    arrival: Endpoint
    means_of_transport: Literal["train", "flixtrain", "bus", "flixbus"]
    operator_id: str


class Price(BaseModel):
    total: float = Field(gt=0, allow_inf_nan=False)
    total_with_platform_fee: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_total(self) -> Self:
        if self.total_with_platform_fee < self.total:
            raise ValueError("fee-inclusive price is less than ticket price")
        return self


class Offer(BaseModel):
    departure: Endpoint
    arrival: Endpoint
    duration: dict[str, int]
    price: Price
    legs: list[Leg] = Field(min_length=1)


class FlixProvider:
    """Fetches geographically validated, fee-inclusive FlixBus and FlixTrain quotes."""

    FLIX_AUTOCOMPLETE = "https://global.api.flixbus.com/search/autocomplete/cities"
    FLIX_SEARCH = "https://global.api.flixbus.com/search/service/v4/search"
    MIN_REQUEST_INTERVAL = 1.0

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session(impersonate="chrome124")
        self._city_cache: dict[str, FlixCity] = {}
        self._last_request = 0.0

    def _get_json(self, url: str, params: dict) -> dict | list:
        time.sleep(max(0.0, self.MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (RequestException, ValueError) as exc:
            raise TransitAPIError(f"Flix request failed at {url}: {exc}") from exc

    def resolve_city(self, query_name: str, anchor: City) -> FlixCity:
        if query_name in self._city_cache:
            return self._city_cache[query_name]

        data = self._get_json(self.FLIX_AUTOCOMPLETE, {"q": anchor.name, "lang": "en"})
        if not isinstance(data, list):
            raise TransitAPIError(f"Unexpected Flix city response for {query_name}")

        aliases = {normalized_name(alias) for alias in anchor.aliases}
        candidates = []
        for raw in data:
            try:
                city = FlixCity.model_validate(raw)
            except ValidationError:
                continue
            offset = distance_km(anchor.latitude, anchor.longitude, city.location.lat, city.location.lon)
            if (city.country.lower() == anchor.country
                    and (normalized_name(city.name) in aliases or offset <= 15)
                    and offset <= 30):
                candidates.append((offset, city))

        if not candidates:
            raise TransitAPIError(f"No geographically verified Flix city for {query_name!r}")

        city = min(candidates, key=lambda c: c[0])[1]
        self._city_cache[query_name] = city
        return city

    def _parse_offer(
        self, uid: str, raw: dict, data: dict,
        origin: str, destination: str, orig_anchor: City, dest_anchor: City, quoted_at: datetime,
    ) -> ConnectionResult:
        offer = Offer.model_validate(raw)
        src, dest = self._city_cache[origin], self._city_cache[destination]
        if offer.departure.city_id != src.id or offer.arrival.city_id != dest.id:
            raise ValueError("offer endpoints do not match requested city IDs")
        if offer.legs[0].departure != offer.departure or offer.legs[-1].arrival != offer.arrival:
            raise ValueError("leg endpoints do not match the complete journey")

        previous = None
        modes, operators = set(), []
        for leg in offer.legs:
            if leg.arrival.date <= leg.departure.date:
                raise ValueError("leg arrival is not after departure")
            if previous and (leg.departure.date < previous.date or leg.departure.city_id != previous.city_id):
                raise ValueError("legs overlap or have a disconnected transfer")
            previous = leg.arrival
            modes.add("train" if leg.means_of_transport in ("train", "flixtrain") else "bus")
            operator = data["operators"][leg.operator_id]["label"]
            if not isinstance(operator, str) or not operator.strip():
                raise ValueError("missing operator label")
            if operator not in operators:
                operators.append(operator)

        elapsed = (offer.arrival.date - offer.departure.date).total_seconds() / 60
        declared = offer.duration["hours"] * 60 + offer.duration["minutes"]
        if abs(declared - elapsed) > 1.0:
            raise ValueError("provider duration disagrees with timestamps")

        est_dist, est_co2 = estimate_route_distance_and_co2(orig_anchor, dest_anchor, modes)
        return ConnectionResult(
            journey_id=uid,
            source_api="flixbus",
            origin_city=origin,
            destination_city=destination,
            departure_time=offer.departure.date,
            arrival_time=offer.arrival.date,
            duration_minutes=round(elapsed),
            stops=len(offer.legs) - 1,
            price_eur=offer.price.total_with_platform_fee,
            is_price_estimated=False,
            quoted_at=quoted_at,
            operators=operators,
            transport_modes=sorted(modes),
            station_src=data["stations"][offer.departure.station_id]["name"],
            station_dest=data["stations"][offer.arrival.station_id]["name"],
            distance_km=est_dist,
            co2_kg=est_co2,
        )

    def fetch_connections(
        self, origin: str, destination: str, orig_anchor: City, dest_anchor: City,
        day: date, start: dt_time, end: dt_time,
    ) -> list[ConnectionResult]:
        src = self.resolve_city(origin, orig_anchor)
        dest = self.resolve_city(destination, dest_anchor)
        data = self._get_json(self.FLIX_SEARCH, {
            "from_city_id": src.id,
            "to_city_id": dest.id,
            "departure_date": day.strftime("%d.%m.%Y"),
            "products": '{"adult":1}',
            "currency": "EUR",
            "locale": "en",
            "search_by": "cities",
            "include_after_midnight_rides": "0",
        })
        if not isinstance(data, dict) or not isinstance(data.get("trips"), list):
            raise TransitAPIError("Unexpected Flix search response (missing trips)")
        if data.get("errors") or data.get("error"):
            raise TransitAPIError("Flix returned an error instead of search results")
        if data.get("currency", "EUR") != "EUR":
            raise TransitAPIError("Flix returned a different currency than EUR")

        quoted_at = datetime.now(timezone.utc)
        results: dict[str, ConnectionResult] = {}
        for trip in data["trips"]:
            offers = trip.get("results")
            if not isinstance(offers, dict):
                continue
            for uid, raw in offers.items():
                if not isinstance(raw, dict) or raw.get("status") != "available":
                    continue
                if raw.get("restrictions", {}).get("sale_restriction"):
                    continue
                if raw.get("available", {}).get("seats") == 0:
                    continue
                if raw.get("price", {}).get("currency", "EUR") != "EUR":
                    continue
                try:
                    result = self._parse_offer(uid, raw, data, origin, destination, orig_anchor, dest_anchor, quoted_at)
                except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    logger.debug("Rejecting Flix offer %s for %s -> %s: %s", uid, origin, destination, exc)
                    continue
                if result.departure_time.date() == day and start <= result.departure_time.time() <= end:
                    results[uid] = result

        return list(results.values())
