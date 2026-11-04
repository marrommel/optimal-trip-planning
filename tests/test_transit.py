import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

for _sub in ("src", "src/helpers", "src/optimizer"): sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))

import data_loader
import config
import decoder
import main
import nsga2
from models import Chromosome, Connection, Phenotype
from api.train_api import CITIES, ConnectionResult, EuropeTransitScraper, FlixCity, TransitAPIError


DAY = "2026-12-10"


def city(name):
    anchor = CITIES[name]
    return {
        "id": name, "name": name, "country": anchor.country,
        "location": {"lat": anchor.latitude, "lon": anchor.longitude},
    }


def endpoint(name, at):
    return {"date": at, "city_id": name, "station_id": name + "-station"}


def response(origin="Berlin", destination="Hamburg",
             departure=DAY + "T08:07:00+01:00", arrival=DAY + "T10:21:00+01:00",
             mode="train"):
    dep, arr = endpoint(origin, departure), endpoint(destination, arrival)
    elapsed = int((datetime.fromisoformat(arrival) - datetime.fromisoformat(departure)).total_seconds() / 60)
    hours, minutes = divmod(elapsed, 60)
    offer = {
        "status": "available", "departure": dep, "arrival": arr,
        "duration": {"hours": hours, "minutes": minutes},
        "price": {"total": 6.49, "total_with_platform_fee": 7.48},
        "legs": [{"departure": dep, "arrival": arr,
                  "means_of_transport": mode, "operator_id": mode}],
    }
    return {
        "trips": [{"results": {"offer-1": offer}}],
        "operators": {"train": {"label": "FlixTrain"}, "bus": {"label": "FlixBus"}},
        "stations": {
            origin + "-station": {"name": origin + " Central Station"},
            destination + "-station": {"name": destination + " Central Station"},
        },
    }


def client_for(data, origin="Berlin", destination="Hamburg"):
    client = EuropeTransitScraper()
    client._city_cache = {
        name: FlixCity.model_validate(city(name)) for name in (origin, destination)
    }
    client._get_json = Mock(return_value=data)
    return client


def quote():
    return client_for(response()).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)[0]


class StationTests(unittest.TestCase):
    def test_wrong_swiss_match_is_rejected(self):
        client = EuropeTransitScraper()
        client._get_json = Mock(return_value=[
            {"id": "8503083", "name": "Zürich, Römerhof", "country": "ch",
             "location": {"lat": 47.36809, "lon": 8.56028}},
        ])
        with self.assertRaisesRegex(TransitAPIError, "geographically verified"):
            client.resolve_city("Rome")

    def test_checks_all_candidates_not_first(self):
        client = EuropeTransitScraper()
        wrong = city("Rome")
        wrong["location"] = city("Berlin")["location"]
        client._get_json = Mock(return_value=[wrong, city("Rome")])
        self.assertAlmostEqual(client.resolve_city("Rome").location.lat, 41.9028)
        client.resolve_city("Rome")
        client._get_json.assert_called_once()

    def test_airport_or_missing_coordinates_is_not_city(self):
        client = EuropeTransitScraper()
        airport = city("Rome")
        airport["name"] = "Rome Fiumicino Airport"
        malformed = city("Rome")
        del malformed["location"]
        client._get_json = Mock(return_value=[airport, malformed])
        with self.assertLogs("api.train_api", level="WARNING"):
            with self.assertRaises(TransitAPIError):
                client.resolve_city("Rome")

    def test_unknown_city_fails_without_network(self):
        client = EuropeTransitScraper()
        client._get_json = Mock()
        with self.assertRaisesRegex(ValueError, "validation anchor"):
            client.resolve_city("Unknown")
        client._get_json.assert_not_called()


class OfferTests(unittest.TestCase):
    def test_actual_price_station_and_minutes(self):
        item = quote()
        self.assertEqual(item.price_eur, 7.48)
        self.assertFalse(item.is_price_estimated)
        self.assertEqual(item.duration_minutes, 134)
        self.assertEqual(item.duration_formatted, "2h 14m")
        self.assertEqual(item.station_src, "Berlin Central Station")
        self.assertEqual(item.departure_time.minute, 7)
        self.assertEqual(item.mode, "train")
        self.assertTrue(item.is_co2_estimated)

    def test_currency_and_one_adult_sent_to_provider(self):
        client = client_for(response())
        client.get_cheapest_in_timespan("Berlin", "Hamburg", DAY)
        params = client._get_json.call_args.args[1]
        self.assertEqual(params["currency"], "EUR")
        self.assertEqual(json.loads(params["products"]), {"adult": 1})

    def test_prices_must_be_complete_positive_finite_eur(self):
        for price in ({}, {"total": 12}, {"total": 6, "total_with_platform_fee": 0},
                      {"total": 6, "total_with_platform_fee": float("nan")},
                      {"total": 6, "total_with_platform_fee": 5},
                      {"total": 6, "total_with_platform_fee": 7, "currency": "USD"}):
            with self.subTest(price=price):
                data = response()
                data["trips"][0]["results"]["offer-1"]["price"] = price
                with self.assertLogs("api.train_api", level="WARNING"):
                    with self.assertRaises(TransitAPIError):
                        client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)

    def test_no_timezone_no_duration_or_wrong_endpoint_is_rejected(self):
        for change in ("timezone", "duration", "endpoint", "station", "operator"):
            with self.subTest(change=change):
                data = response()
                raw = data["trips"][0]["results"]["offer-1"]
                if change == "timezone":
                    raw["departure"]["date"] = DAY + "T08:07:00"
                elif change == "duration":
                    raw["duration"]["hours"] = 1
                elif change == "endpoint":
                    raw["arrival"]["city_id"] = "Rome"
                elif change == "station":
                    data["stations"] = {}
                else:
                    data["operators"] = {}
                with self.assertLogs("api.train_api", level="WARNING"):
                    with self.assertRaises(TransitAPIError):
                        client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)

    def test_timezone_crossing_and_overnight_duration(self):
        data = response("London", "Paris", DAY + "T22:00:00+00:00",
                        "2026-12-11T08:00:00+01:00", "bus")
        item = client_for(data, "London", "Paris").get_cheapest_in_timespan("London", "Paris", DAY)[0]
        self.assertEqual(item.duration_minutes, 540)
        self.assertEqual(item.arrival_time.day, 11)

    def test_daylight_saving_offset_change(self):
        data = response(departure="2026-10-25T01:30:00+02:00", arrival="2026-10-25T03:30:00+01:00")
        item = client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", "2026-10-25", "00:00")[0]
        self.assertEqual(item.duration_minutes, 180)

    def test_mixed_journey_is_not_pure_train(self):
        data = response(arrival=DAY + "T13:07:00+01:00")
        raw = data["trips"][0]["results"]["offer-1"]
        transfer_arr = endpoint("transfer", DAY + "T09:30:00+01:00")
        transfer_dep = endpoint("transfer", DAY + "T10:00:00+01:00")
        raw["legs"] = [
            {"departure": raw["departure"], "arrival": transfer_arr,
             "means_of_transport": "train", "operator_id": "train"},
            {"departure": transfer_dep, "arrival": raw["arrival"],
             "means_of_transport": "bus", "operator_id": "bus"},
        ]
        item = client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)[0]
        self.assertEqual(item.mode, "train+bus")
        self.assertEqual(item.operators, ["FlixTrain", "FlixBus"])
        self.assertEqual(item.stops, 1)
        raw["legs"][1]["departure"]["date"] = DAY + "T09:00:00+01:00"
        with self.assertLogs("api.train_api", level="WARNING"):
            with self.assertRaises(TransitAPIError):
                client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)

    def test_unavailable_and_out_of_window_offers_excluded(self):
        for change in ("sold_out", "restriction", "seats", "time", "date"):
            with self.subTest(change=change):
                data = response()
                raw = data["trips"][0]["results"]["offer-1"]
                if change == "sold_out":
                    raw["status"] = "sold_out"
                elif change == "restriction":
                    raw["restrictions"] = {"sale_restriction": True}
                elif change == "seats":
                    raw["available"] = {"seats": 0}
                client = client_for(data)
                with self.assertLogs("api.train_api", level="WARNING"):
                    result = client.get_cheapest_in_timespan(
                        "Berlin", "Hamburg", "2026-12-11" if change == "date" else DAY,
                        "09:00" if change == "time" else "05:00")
                self.assertEqual(result, [])

    def test_slow_trips_not_rejected_but_impossible_speed_is(self):
        data = response(arrival="2026-12-11T10:21:00+01:00")
        self.assertEqual(len(client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)), 1)
        data = response(arrival=DAY + "T08:08:00+01:00")
        with self.assertLogs("api.train_api", level="WARNING"):
            with self.assertRaises(TransitAPIError):
                client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)

    def test_no_results_and_provider_error_are_distinct(self):
        with self.assertLogs("api.train_api", level="WARNING"):
            self.assertEqual(client_for({"trips": []}).get_cheapest_in_timespan("Berlin", "Hamburg", DAY), [])
        for data in ({}, {"trips": [], "error": "failed"}, {"trips": [], "currency": "USD"}):
            with self.assertRaises(TransitAPIError):
                client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)

    def test_network_failure_is_not_an_empty_search(self):
        from curl_cffi.requests.exceptions import RequestException
        client = EuropeTransitScraper()
        client.MIN_REQUEST_INTERVAL = 0
        client.session.get = Mock(side_effect=RequestException("HTTP 429"))
        with self.assertRaisesRegex(TransitAPIError, "HTTP 429"):
            client.resolve_city("Rome")
        client.session.get.assert_called_once()

    def test_preserves_multiple_departures_price_sorted(self):
        data = response()
        second = copy.deepcopy(data["trips"][0]["results"]["offer-1"])
        second["departure"]["date"] = DAY + "T09:07:00+01:00"
        second["arrival"]["date"] = DAY + "T11:21:00+01:00"
        second["price"] = {"total": 5, "total_with_platform_fee": 6}
        data["trips"][0]["results"]["offer-2"] = second
        items = client_for(data).get_cheapest_in_timespan("Berlin", "Hamburg", DAY)
        self.assertEqual([q.price_eur for q in items], [6, 7.48])
        self.assertEqual([q.departure_time.hour for q in items], [9, 8])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache_dir = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            data_loader, CACHE_DIR=self.cache_dir, CITIES_LIST=["Berlin", "Hamburg"],
            CITIES={0: "Berlin", 1: "Hamburg"}, TRAVEL_RANGE=[DAY, DAY],
        ))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def cache(self, **changes):
        values = dict(cities=["Berlin", "Hamburg"], travel_range=(DAY, DAY),
                      fetched_at=datetime.now(timezone.utc), connections=[quote()])
        values.update(changes)
        cache = data_loader.TransitCache(**values)
        (self.cache_dir / "transit_quotes.json").write_text(cache.model_dump_json(), encoding="utf-8")

    def test_valid_cache_avoids_network_and_preserves_quote(self):
        self.cache()
        with patch.object(data_loader, "EuropeTransitScraper", wraps=EuropeTransitScraper) as client:
            frame = data_loader.fetch_and_cache_transit()
            client.assert_not_called()
        self.assertEqual(frame.iloc[0]["price"], 7.48)
        self.assertEqual(frame.iloc[0]["dep_time"], "08:07")
        self.assertEqual(frame.iloc[0]["station_src"], "Berlin Central Station")
        self.assertEqual(frame.iloc[0]["day"], 0)

    def test_cache_and_quotes_do_not_expire(self):
        old = datetime.now(timezone.utc) - timedelta(days=3650)
        for connections in ([quote().model_copy(update={"quoted_at": old})], []):
            with self.subTest(empty=not connections):
                self.cache(fetched_at=old, connections=connections)
                with patch.object(data_loader, "EuropeTransitScraper") as client:
                    frame = data_loader.fetch_and_cache_transit()
                client.assert_not_called()
                self.assertEqual(len(frame), len(connections))
                if connections:
                    self.assertEqual(datetime.fromisoformat(frame.iloc[0]["quoted_at"]), old)

    def test_config_mismatch_and_unpriced_caches_refetched(self):
        for change in ("config", "date", "schema", "estimated", "unknown_city"):
            with self.subTest(change=change):
                self.cache()
                path = self.cache_dir / "transit_quotes.json"
                content = json.loads(path.read_text(encoding="utf-8"))
                if change == "config":
                    content["cities"] = ["Hamburg", "Berlin"]
                elif change == "date":
                    content["travel_range"] = ["2026-12-11", "2026-12-11"]
                elif change == "schema":
                    content["schema_version"] = 1
                elif change == "estimated":
                    content["connections"][0]["is_price_estimated"] = True
                elif change == "unknown_city":
                    content["connections"][0]["origin_city"] = "Fake city"
                path.write_text(json.dumps(content), encoding="utf-8")
                with patch.object(EuropeTransitScraper, "get_cheapest_in_timespan", return_value=[]) as search:
                    with self.assertLogs("data_loader", level="WARNING"):
                        frame = data_loader.fetch_and_cache_transit()
                    self.assertEqual(search.call_count, 2)
                    self.assertTrue(frame.empty)
                    self.assertEqual(list(frame.columns), data_loader.TRANSIT_COLUMNS)

    def test_poisoned_legacy_csv_ignored_not_deleted(self):
        legacy = self.cache_dir / "transit_cache.csv"
        legacy.write_text("price,duration\n71.95,4.3\n", encoding="utf-8")
        with patch.object(EuropeTransitScraper, "get_cheapest_in_timespan", side_effect=[[quote()], []]):
            with self.assertLogs("data_loader", level="WARNING"):
                frame = data_loader.fetch_and_cache_transit()
        self.assertTrue(legacy.exists())
        self.assertEqual(frame.iloc[0]["price"], 7.48)

    def test_provider_failure_never_writes_success_cache(self):
        with patch.object(EuropeTransitScraper, "get_cheapest_in_timespan",
                          side_effect=[[quote()], TransitAPIError("provider unavailable")]):
            with self.assertRaises(TransitAPIError):
                data_loader.fetch_and_cache_transit()
        self.assertFalse((self.cache_dir / "transit_quotes.json").exists())

    def test_unpriced_airport_transit_is_not_fabricated(self):
        client = Mock()
        client.get_cheapest_in_timespan.return_value = []
        with patch.dict(data_loader._TRANSIT_API_CACHE, clear=True):
            with self.assertRaises(TransitAPIError):
                data_loader.get_airport_transit("FRA", DAY, client)
        with self.assertRaises(TransitAPIError):
            data_loader.get_airport_transit("BAD", DAY, client)

    def test_all_quotes_loaded_and_database_references_preserved(self):
        first = quote()
        second = first.model_copy(update={"journey_id": "second", "price_eur": 9.0})
        frame = data_loader._transit_frame([first, second])
        with patch.object(data_loader, "fetch_and_cache_transit", return_value=frame), \
                patch.object(data_loader, "fetch_and_cache_flights", return_value=frame.iloc[:0]), \
                patch.object(data_loader, "load_hotels", return_value=[]):
            original = list(data_loader.CONNECTIONS_DB)
            self.addCleanup(lambda: data_loader.CONNECTIONS_DB.__setitem__(slice(None), original))
            with patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True):
                data_loader.initialize_database()
                self.assertIs(decoder.CONNECTIONS_BY_ROUTE, data_loader.CONNECTIONS_BY_ROUTE)
                self.assertEqual(len(decoder.CONNECTIONS_BY_ROUTE[(0, 0, 1)]), 2)
        self.assertEqual(len(data_loader.CONNECTIONS_DB), 2)
        conn = data_loader.CONNECTIONS_DB[0]
        self.assertEqual(conn.fare_source, "flixbus")
        self.assertEqual(conn.arrival_time.minute, 21)
        ind = SimpleNamespace(objectives=(conn.cost, 0, conn.co2),
                              phenotype=Phenotype([0, 1], [], [conn], []))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main.print_itinerary("Test", ind)
        self.assertIn("08:07", out.getvalue())
        self.assertNotIn("fees included", out.getvalue())
        self.assertNotIn("08:00", out.getvalue())

    def test_mixed_mode_remains_available_as_fallback(self):
        conn = Connection(0, 0, 1, "train+bus", 8, 5, 40, 10,
                          departure_time=datetime.fromisoformat(DAY + "T08:00+01:00"),
                          arrival_time=datetime.fromisoformat(DAY + "T13:00+01:00"))
        chromosome = Chromosome([1], [2], [8, 8], [0, 0], [0, 0])
        with patch.dict(decoder.CONNECTIONS_BY_ROUTE, {(0, 0, 1): [conn]}, clear=True):
            result = decoder.decode(chromosome)
        self.assertEqual(result.connections[0].mode, "train+bus")

    def test_optimizer_entrypoint_initializes_and_uses_quotes(self):
        outbound = Connection(0, 0, 1, "train", 8 + 7 / 60, 2, 7.48, 4,
                              fare_source="flixbus", quoted_at="2026-10-30T12:00:00+00:00",
                              departure_time=datetime.fromisoformat(DAY + "T08:07+01:00"),
                              arrival_time=datetime.fromisoformat(DAY + "T10:07+01:00"))
        inbound = Connection(2, 1, 0, "train", 10, 2, 7.48, 4,
                             fare_source="flixbus", quoted_at="2026-10-30T12:00:00+00:00",
                             departure_time=datetime.fromisoformat("2026-12-12T10:00+01:00"),
                             arrival_time=datetime.fromisoformat("2026-12-12T12:00+01:00"))
        original = list(data_loader.CONNECTIONS_DB)
        self.addCleanup(lambda: data_loader.CONNECTIONS_DB.__setitem__(slice(None), original))
        out = io.StringIO()
        with patch.object(data_loader, "load_connections", return_value=[outbound, inbound]), \
                patch.object(data_loader, "load_hotels", return_value=[]), \
                patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True), \
                patch.multiple(nsga2, NUM_DESTINATIONS=1, TOTAL_AVAILABLE_DESTINATIONS=1), \
                patch.multiple(config, NUM_DESTINATIONS=1, TOTAL_DAYS=3, OPTIMIZE_HOTELS=False,
                               TRAVEL_RANGE=["2026-12-10", "2026-12-12"]), \
                patch.multiple(main, POPULATION_SIZE=4, MAX_GENERATIONS=1), \
                contextlib.redirect_stdout(out):
            main.main()
        self.assertIn("FINAL RECOMMENDED ITINERARIES", out.getvalue())
        self.assertIn("08:07", out.getvalue())
        self.assertNotIn("Fare quote:", out.getvalue())

    def test_legacy_flights_without_timestamps_are_refetched_not_migrated(self):
        (self.cache_dir / "flights_cache.csv").write_text(
            "price,transit_mode,transit_cost,transit_hrs\n"
            "10,Local Transit,0,0\n20,,0,0\n30,Fallback Transit,70,4\n",
            encoding="utf-8",
        )
        with patch.object(data_loader, "search_flights", return_value=[]), \
                patch.object(data_loader.time, "sleep"), \
                self.assertLogs("data_loader", level="WARNING"):
            frame = data_loader.fetch_and_cache_flights()
        self.assertTrue(frame.empty)
        self.assertTrue((self.cache_dir / "flights_cache.csv").exists())
        self.assertTrue((self.cache_dir / "flight_quotes.json").exists())


if __name__ == "__main__":
    unittest.main()
