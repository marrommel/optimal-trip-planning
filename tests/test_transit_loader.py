import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for sub in ("src", "src/helpers", "src/optimizer"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / sub))

import config
import data_loader
import objectives
from api.train_api import ConnectionResult, EuropeTransitScraper, TransitAPIError
from api.train_helpers import db_provider, flix_provider, train_models
from models import Phenotype


DAY = "2026-12-10"


def quote(source="deutsche_bahn", modes=("train",), co2=83.125, **changes):
    origin, destination = train_models.CITIES["Stuttgart"], train_models.CITIES["Paris"]
    distance, _ = train_models.estimate_route_distance_and_co2(origin, destination, modes)
    values = dict(
        journey_id=source + "-" + "-".join(modes), source_api=source,
        origin_city="Stuttgart", destination_city="Paris",
        departure_time=DAY + "T08:07:00+01:00", arrival_time=DAY + "T12:07:00+01:00",
        duration_minutes=240, stops=len(modes) - 1, price_eur=60,
        is_price_estimated=source == "deutsche_bahn",
        quoted_at=datetime.now(timezone.utc), operators=["Fixture operator"],
        transport_modes=list(modes), station_src="Stuttgart Hbf", station_dest="Paris Est",
        distance_km=distance, co2_kg=co2,
    )
    values.update(changes)
    return ConnectionResult(**values)


class TransitLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache_dir = Path(self.temp.name) / "cache"
        self.cache_file = self.cache_dir / "transit_quotes.json"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            data_loader, CACHE_DIR=self.cache_dir, CITIES_LIST=["Stuttgart", "Paris"],
            CITIES={0: "Stuttgart", 1: "Paris"}, TRAVEL_RANGE=[DAY, DAY],
        ))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def write_cache(self, quotes):
        cache = data_loader.TransitCache(
            cities=["Stuttgart", "Paris"], travel_range=(DAY, DAY),
            fetched_at=datetime.now(timezone.utc), connections=quotes,
        )
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file.write_text(cache.model_dump_json(), encoding="utf-8")

    def forbid_emission_estimation(self):
        for module in (train_models, db_provider, flix_provider):
            self.stack.enter_context(patch.object(
                module, "estimate_route_distance_and_co2",
                side_effect=AssertionError("CO2 must come from the cached API response"),
            ))

    def test_both_sources_survive_fetch_cache_load_and_objective_evaluation(self):
        quotes = [
            quote(),
            quote("flixbus", ("bus",), 7.375, price_eur=20),
            quote("flixbus", ("train", "bus"), 0.0, price_eur=30),
        ]
        self.forbid_emission_estimation()
        with patch.object(EuropeTransitScraper, "fetch_db", side_effect=[[quotes[0]], []]) as db, \
                patch.object(EuropeTransitScraper, "fetch_flix", side_effect=[quotes[1:], []]) as flix:
            frame = data_loader.fetch_and_cache_transit()
        self.assertEqual(db.call_count, 2)
        self.assertEqual(flix.call_count, 2)
        self.assertEqual(db.call_args_list, flix.call_args_list)
        self.assertEqual(db.call_args_list[0].args, ("Stuttgart", "Paris", DAY, "05:00", "23:59"))
        self.assertEqual(frame["co2"].tolist(), [7.375, 0.0, 83.125])
        self.assertEqual(frame["mode"].tolist(), ["bus", "train+bus", "train"])
        self.assertEqual(frame["is_price_estimated"].tolist(), [False, False, True])
        self.assertEqual(frame["is_co2_estimated"].tolist(), [True, True, True])
        saved = json.loads(self.cache_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["schema_version"], 3)
        self.assertEqual(saved["emissions_source"], "transit-api")
        self.assertEqual([q["co2_kg"] for q in saved["connections"]], [7.375, 0.0, 83.125])

        with patch.object(data_loader, "EuropeTransitScraper",
                          side_effect=AssertionError("A valid cache must not fetch again")), \
                patch.object(data_loader, "fetch_and_cache_flights", return_value=frame.iloc[:0]), \
                patch.multiple(config, NUM_TRAVELLERS=3, OPTIMIZE_HOTELS=False):
            connections = data_loader.load_connections()
            expected = [7.375 * 3, 0.0, 83.125 * 3]
            self.assertEqual([conn.co2 for conn in connections], expected)
            self.assertEqual([conn.cost for conn in connections], [60, 90, 180])
            for conn in connections:
                data_loader.validate_connection(conn)
                self.assertIn("cached estimate", conn.co2_method)
                self.assertEqual(objectives.connection_objectives(conn)[2], conn.co2)
            phenotype = Phenotype([0, 1], [], connections, [])
            for _ in range(3):
                self.assertEqual(objectives.objective_values(phenotype)[2], sum(expected))
        self.assertEqual(json.loads(self.cache_file.read_text(encoding="utf-8")), saved)

    def test_old_schema_and_missing_or_invalid_emissions_trigger_refresh(self):
        fresh = quote(co2=12.345)
        for corruption in ("old_schema", "missing_schema", "missing_co2", "negative", "nan"):
            with self.subTest(corruption=corruption):
                self.write_cache([quote()])
                saved = json.loads(self.cache_file.read_text(encoding="utf-8"))
                if corruption == "old_schema":
                    saved["schema_version"] = 2
                elif corruption == "missing_schema":
                    del saved["schema_version"]
                elif corruption == "missing_co2":
                    del saved["connections"][0]["co2_kg"]
                else:
                    saved["connections"][0]["co2_kg"] = -1 if corruption == "negative" else "NaN"
                self.cache_file.write_text(json.dumps(saved), encoding="utf-8")
                with patch.object(EuropeTransitScraper, "get_cheapest_in_timespan",
                                  side_effect=[[fresh], []]) as search, \
                        self.assertLogs("data_loader", level="WARNING"):
                    frame = data_loader.fetch_and_cache_transit()
                self.assertEqual(search.call_count, 2)
                self.assertEqual(frame["co2"].tolist(), [12.345])
                saved = json.loads(self.cache_file.read_text(encoding="utf-8"))
                self.assertEqual(saved["connections"][0]["co2_kg"], 12.345)

    def test_empty_dual_source_cache_is_reused(self):
        self.write_cache([])
        with patch.object(data_loader, "EuropeTransitScraper") as client:
            frame = data_loader.fetch_and_cache_transit()
        client.assert_not_called()
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), data_loader.TRANSIT_COLUMNS)

    def test_failed_refresh_does_not_replace_existing_cache(self):
        self.write_cache([quote()])
        saved = json.loads(self.cache_file.read_text(encoding="utf-8"))
        saved["schema_version"] = 2
        self.cache_file.write_text(json.dumps(saved), encoding="utf-8")
        before = self.cache_file.read_bytes()
        with patch.object(EuropeTransitScraper, "get_cheapest_in_timespan",
                          side_effect=TransitAPIError("unavailable")), \
                self.assertLogs("data_loader", level="WARNING"):
            with self.assertRaises(TransitAPIError):
                data_loader.fetch_and_cache_transit()
        self.assertEqual(self.cache_file.read_bytes(), before)

    def test_rail_provider_timezone_is_converted_to_departure_city(self):
        origin, destination = train_models.CITIES["London"], train_models.CITIES["Paris"]
        distance, _ = train_models.estimate_route_distance_and_co2(origin, destination, ("train",))
        rail = quote(
            origin_city="London", station_src="London St Pancras", distance_km=distance,
            departure_time=DAY + "T10:07:00+01:00", arrival_time=DAY + "T14:07:00+01:00",
        )
        self.forbid_emission_estimation()
        with patch.multiple(data_loader, CITIES_LIST=["London", "Paris"], CITIES={0: "London", 1: "Paris"}):
            frame = data_loader._transit_frame([rail])
            self.assertEqual(frame.iloc[0]["dep_time"], "09:07")
            with patch.object(data_loader, "fetch_and_cache_transit", return_value=frame), \
                    patch.object(data_loader, "fetch_and_cache_flights", return_value=frame.iloc[:0]):
                connection = data_loader.load_connections()[0]
            data_loader.validate_connection(connection)
            self.assertEqual(connection.co2, 83.125 * config.NUM_TRAVELLERS)


if __name__ == "__main__":
    unittest.main()
