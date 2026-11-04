import logging
from datetime import datetime

from curl_cffi import requests

from .api_helpers.progress import ProgressLogger
from .api_helpers.db_provider import DeutscheBahnProvider
from .api_helpers.flix_provider import FlixProvider
from .api_helpers.train_models import (
    CITIES,
    ConnectionResult,
    TransitAPIError,
    resolve_feasible_route,
)

logger = logging.getLogger(__name__)
_db_progress = ProgressLogger(logger, "DB trains", "connections")
_flix_progress = ProgressLogger(logger, "Flix", "connections")


class EuropeTransitScraper:
    """Unified European train & bus scraper with dedicated methods for DB, Flix, or both."""

    def __init__(self):
        self.session = requests.Session(impersonate="chrome124")
        self.db_provider = DeutscheBahnProvider(self.session)
        self.flix_provider = FlixProvider(self.session)

    @staticmethod
    def _parse_window(date_str: str, start_time: str, end_time: str):
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
        start = datetime.strptime(start_time, "%H:%M").time()
        end = datetime.strptime(end_time, "%H:%M").time()
        if start > end:
            raise ValueError("start_time must not be after end_time")
        return day, start, end

    def fetch_db(
        self,
        origin: str,
        destination: str,
        date_str: str,
        start_time: str = "05:00",
        end_time: str = "23:59",
    ) -> list[ConnectionResult]:
        """Fetch ONLY Deutsche Bahn / pan-European rail connections."""
        day, start, end = self._parse_window(date_str, start_time, end_time)
        feasible = resolve_feasible_route(origin, destination, CITIES)
        if feasible is None:
            return []
        orig_anchor, dest_anchor = feasible
        try:
            quotes = self.db_provider.fetch_connections(origin, destination, orig_anchor, dest_anchor, day, start, end)
            _db_progress.record(len(quotes))
            return sorted(quotes, key=lambda r: (r.price_eur, r.duration_minutes))
        except TransitAPIError as exc:
            logger.warning("DB provider failed for %s -> %s on %s: %s", origin, destination, date_str, exc)
            return []

    def fetch_flix(
        self,
        origin: str,
        destination: str,
        date_str: str,
        start_time: str = "05:00",
        end_time: str = "23:59",
    ) -> list[ConnectionResult]:
        """Fetch ONLY FlixBus / FlixTrain connections."""
        day, start, end = self._parse_window(date_str, start_time, end_time)
        feasible = resolve_feasible_route(origin, destination, CITIES)
        if feasible is None:
            return []
        orig_anchor, dest_anchor = feasible
        try:
            quotes = self.flix_provider.fetch_connections(origin, destination, orig_anchor, dest_anchor, day, start, end)
            _flix_progress.record(len(quotes))
            return sorted(quotes, key=lambda r: (r.price_eur, r.duration_minutes))
        except TransitAPIError as exc:
            logger.warning("Flix provider failed for %s -> %s on %s: %s", origin, destination, date_str, exc)
            return []

    def fetch_both(
        self,
        origin: str,
        destination: str,
        date_str: str,
        start_time: str = "05:00",
        end_time: str = "23:59",
    ) -> tuple[list[ConnectionResult], list[ConnectionResult]]:
        """Return a tuple of (db_quotes, flix_quotes) so you can inspect or cache both sources separately."""
        db_quotes = self.fetch_db(origin, destination, date_str, start_time, end_time)
        flix_quotes = self.fetch_flix(origin, destination, date_str, start_time, end_time)
        return db_quotes, flix_quotes

    def get_cheapest_in_timespan(
        self,
        origin: str,
        destination: str,
        date_str: str,
        start_time: str = "05:00",
        end_time: str = "23:59",
    ) -> list[ConnectionResult]:
        """Return merged list of both DB and Flix connections sorted by price."""
        db_quotes, flix_quotes = self.fetch_both(origin, destination, date_str, start_time, end_time)
        return sorted(
            db_quotes + flix_quotes,
            key=lambda r: (r.price_eur, r.duration_minutes, r.is_price_estimated),
        )


def _print_quotes(title: str, quotes: list[ConnectionResult], limit: int = 5) -> None:
    print(f"\n=== {title} (showing {min(len(quotes), limit)} of {len(quotes)}) ===")
    if not quotes:
        print("  [No connections found]")
        return
    for q in quotes[:limit]:
        tag = "est. rail tariff" if q.is_price_estimated else "quoted, fees incl."
        print(
            f"  EUR {q.price_eur:6.2f} ({tag}) | [{q.source_api}] | "
            f"{q.departure_time.isoformat()} -> {q.arrival_time.isoformat()} | "
            f"{q.duration_formatted} ({q.stops} stops) | {q.mode} ({', '.join(q.operators)}) | "
            f"{q.station_src} -> {q.station_dest} | CO2 ~{q.co2_kg:.2f} kg"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ORIGIN = "stuttgart"
    DESTINATION = "berlin"
    START_DATE = "2026-12-06"
    END_DATE = "2026-12-12"

    scraper = EuropeTransitScraper()

    # Fetch connection for a single day
    print(f"\n=== Testing {ORIGIN} -> {DESTINATION} on {START_DATE} ===")
    db_quotes, flix_quotes = scraper.fetch_both(ORIGIN, DESTINATION, START_DATE)
    _print_quotes("DEUTSCHE BAHN / EUROPEAN RAIL", db_quotes, limit=5)
    _print_quotes("FLIXBUS / FLIXTRAIN", flix_quotes, limit=5)
