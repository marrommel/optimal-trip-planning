import logging
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException
from pydantic import ValidationError

from .train_models import (
    City,
    ConnectionResult,
    TransitAPIError,
    estimate_route_distance_and_co2,
)

logger = logging.getLogger(__name__)
BERLIN_TZ = ZoneInfo("Europe/Berlin")

HIGH_SPEED_CATEGORIES = {
    "ICE", "TGV", "FR", "FRECCIAROSSA", "RJ", "RJX", "ECE", "EC", "IC",
    "EST", "EUROSTAR", "THA", "AVE", "EN", "NJ", "EUROCITY", "INTERCITY", "LYRIA",
}

URBAN_TRANSIT_MODES = {
    "WALK", "FOOT", "TRANSFER", "SUBWAY", "TRAM", "METRO", "UBAHN", "FUSSWEG", "UEBERGANG",
}


class DeutscheBahnProvider:
    """Pan-European Rail Provider powered by Transitous."""

    TRANSITOUS_PLAN = "https://api.transitous.org/api/v1/plan"
    MIN_REQUEST_INTERVAL = 1.0
    USER_AGENT = "OptimalTripPlanningResearch/1.0 (https://github.com/optimal-trip-planning; academic-research)"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session(impersonate="chrome124")
        self._last_request = 0.0

    def _throttle(self) -> None:
        time.sleep(max(0.0, self.MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()

    @staticmethod
    def _active_timetable_day(target_day: date) -> tuple[date, timedelta]:
        """Map dates beyond the active European rail timetable window onto the same weekday ~3 weeks out."""
        today = datetime.now(BERLIN_TZ).date()
        max_safe_date = min(date(today.year, 12, 10), today + timedelta(days=60))
        if today <= target_day <= max_safe_date:
            return target_day, timedelta(0)

        anchor = today + timedelta(days=21)
        days_ahead = (target_day.weekday() - anchor.weekday()) % 7
        projected_day = anchor + timedelta(days=days_ahead)
        return projected_day, (target_day - projected_day)

    @staticmethod
    def _parse_iso_datetime(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(BERLIN_TZ) if dt.tzinfo is not None else dt.replace(tzinfo=BERLIN_TZ)

    @staticmethod
    def _estimate_rail_price(est_distance_km: float, categories: list[str]) -> float:
        high_speed_legs = sum(
            1 for cat in categories
            if any(hs in cat.upper() for hs in HIGH_SPEED_CATEGORIES)
        )
        base_km_rate = 0.11 if high_speed_legs else 0.085
        reservation_surcharge = high_speed_legs * 10.0
        return round(18.0 + (est_distance_km * base_km_rate) + reservation_surcharge, 2)

    def fetch_connections(
        self, origin: str, destination: str, orig_anchor: City, dest_anchor: City,
        day: date, start: dt_time, end: dt_time,
    ) -> list[ConnectionResult]:
        quoted_at = datetime.now(timezone.utc)
        query_day, day_shift = self._active_timetable_day(day)
        if day_shift:
            logger.info(
                "Date %s is beyond active European rail timetable; projecting weekday from %s",
                day.isoformat(), query_day.isoformat(),
            )

        query_dt = datetime.combine(query_day, max(start, dt_time(6, 30)), tzinfo=timezone.utc)
        self._throttle()
        try:
            resp = self.session.get(
                self.TRANSITOUS_PLAN,
                params={
                    "fromPlace": f"{orig_anchor.latitude},{orig_anchor.longitude}",
                    "toPlace": f"{dest_anchor.latitude},{dest_anchor.longitude}",
                    "time": query_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "numItineraries": 10,
                },
                headers={"User-Agent": self.USER_AGENT, "Accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (RequestException, ValueError) as exc:
            raise TransitAPIError(f"Transitous rail API failed: {exc}") from exc

        results: dict[str, ConnectionResult] = {}
        for itin in data.get("itineraries", []):
            try:
                raw_legs = itin.get("legs") or []
                intercity_legs = [
                    leg for leg in raw_legs
                    if (leg.get("mode") or "").upper() not in URBAN_TRANSIT_MODES
                ]
                if not intercity_legs:
                    continue

                dep_str = intercity_legs[0].get("startTime") or itin.get("startTime")
                arr_str = intercity_legs[-1].get("endTime") or itin.get("endTime")
                if not dep_str or not arr_str:
                    continue

                dep_dt = self._parse_iso_datetime(dep_str) + day_shift
                arr_dt = self._parse_iso_datetime(arr_str) + day_shift
                elapsed_min = round((arr_dt - dep_dt).total_seconds() / 60)

                categories, operators, modes = [], [], set()
                for leg in intercity_legs:
                    mode_str = (leg.get("mode") or "RAIL").upper()
                    cat = (leg.get("routeShortName") or leg.get("displayName") or mode_str).strip()
                    op = (leg.get("agencyName") or cat).strip()
                    categories.append(cat)
                    if op and op not in operators:
                        operators.append(op)
                    modes.add("bus" if "BUS" in mode_str or "COACH" in mode_str else "train")

                est_dist, est_co2 = estimate_route_distance_and_co2(orig_anchor, dest_anchor, modes)
                price_eur = self._estimate_rail_price(est_dist, categories)

                src_name = (intercity_legs[0].get("from") or {}).get("name") or origin
                dst_name = (intercity_legs[-1].get("to") or {}).get("name") or destination

                res = ConnectionResult(
                    journey_id=f"transitous-{dep_dt.isoformat()}-{arr_dt.isoformat()}",
                    source_api="deutsche_bahn",
                    origin_city=origin,
                    destination_city=destination,
                    departure_time=dep_dt,
                    arrival_time=arr_dt,
                    duration_minutes=elapsed_min,
                    stops=max(0, len(intercity_legs) - 1),
                    price_eur=price_eur,
                    is_price_estimated=True,
                    quoted_at=quoted_at,
                    operators=operators or ["European Rail"],
                    transport_modes=sorted(modes),
                    station_src=src_name,
                    station_dest=dst_name,
                    distance_km=est_dist,
                    co2_kg=est_co2,
                )

                if res.departure_time.date() == day and start <= res.departure_time.time() <= end:
                    key = f"{res.departure_time.isoformat()}_{res.arrival_time.isoformat()}"
                    if key not in results or res.price_eur < results[key].price_eur:
                        results[key] = res
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping Transitous itinerary: %s", exc)
                continue

        return sorted(results.values(), key=lambda r: (r.price_eur, r.duration_minutes))
