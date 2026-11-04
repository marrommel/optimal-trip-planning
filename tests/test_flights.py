import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

for _sub in ("src", "src/helpers", "src/optimizer"): sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))

import config
import data_loader
from api import flights_api
from timing import localize_schedule


def flight_response(recommended=None, other=None, metadata=None):
    if metadata is None:
        metadata = [None, [[], []]]
    payload = [None, None, recommended, other, None, None, None, metadata]
    return '<script class="ds:1">callback({data:' + json.dumps(payload) + ',other:1});</script>'


class FlightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache_dir = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            data_loader, CACHE_DIR=self.cache_dir, CITIES_LIST=["Stuttgart", "Paris"],
            CITIES={0: "Stuttgart", 1: "Paris"}, TRAVEL_RANGE=["2026-12-10", "2026-12-10"],
            HOME_AIRPORTS=["STR"], CITY_AIRPORTS={"Stuttgart": ["STR"], "Paris": ["CDG"]}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def quote(self, **changes):
        values = dict(
            src_city="Stuttgart", dest_city="Paris", src_airport="STR", dest_airport="CDG",
            departure_time="2026-12-10T10:00+01:00", arrival_time="2026-12-10T14:00+01:00",
            quoted_at=datetime.now(timezone.utc), price=100, air_minutes=120, co2_kg=417,
            stops=1, airline="fixture")
        values.update(changes)
        return data_loader.FlightQuote(**values)

    def write_cache(self, quotes=None, **changes):
        values = dict(cities=["Stuttgart", "Paris"], home_airports=["STR"],
                      travel_range=("2026-12-10", "2026-12-10"), fetched_at=datetime.now(timezone.utc),
                      quotes=quotes if quotes is not None else [self.quote()])
        values.update(changes)
        cache = data_loader.FlightCache(**values)
        (self.cache_dir / "flight_quotes.json").write_text(cache.model_dump_json(), encoding="utf-8")

    def test_cache_retains_provider_emissions_independent_of_air_duration(self):
        self.write_cache([self.quote(), self.quote(price=130, air_minutes=60, stops=0, co2_kg=83.125)])
        with patch.object(data_loader, "search_flights") as search:
            frame = data_loader.fetch_and_cache_flights()
        search.assert_not_called()
        self.assertEqual(frame["price"].tolist(), [100, 130])
        self.assertEqual(frame["co2"].tolist(), [417, 83.125])
        with patch.object(data_loader, "fetch_and_cache_flights", return_value=frame), \
                patch.object(data_loader, "fetch_and_cache_transit", return_value=data_loader._transit_frame([])):
            connections = data_loader.load_connections()
        self.assertEqual(connections[0].stops, 1)
        self.assertEqual(connections[0].arrival_time.hour, 14)
        self.assertEqual(connections[0].access_hours, config.AIRPORT_ACCESS_HOURS["STR"] + config.FLIGHT_CHECK_IN_HOURS)
        self.assertEqual(connections[0].cost, 100)
        self.assertEqual(connections[0].fare_source, "google-flights")
        self.assertEqual(connections[0].co2, 417)
        self.assertIn("Google Flights", connections[0].co2_method)
        from objectives import climate_impact
        self.assertEqual(climate_impact(connections[0]), 417)

    def test_cache_and_quotes_do_not_expire(self):
        old = datetime.now(timezone.utc) - timedelta(days=3650)
        for quotes in ([self.quote(quoted_at=old)], []):
            with self.subTest(empty=not quotes):
                self.write_cache(quotes, fetched_at=old)
                with patch.object(data_loader, "search_flights") as search:
                    frame = data_loader.fetch_and_cache_flights()
                search.assert_not_called()
                self.assertEqual(len(frame), len(quotes))
                if quotes:
                    self.assertEqual(datetime.fromisoformat(frame.iloc[0]["quoted_at"]), old)

    def test_config_mismatch_refetch(self):
        for change in ("cities", "airports", "date", "schema"):
            with self.subTest(change=change):
                if change == "cities":
                    self.write_cache(cities=["Paris", "Stuttgart"])
                elif change == "airports":
                    self.write_cache(home_airports=["FKB"])
                elif change == "date":
                    self.write_cache(travel_range=("2026-12-11", "2026-12-11"))
                else:
                    (self.cache_dir / "flight_quotes.json").write_text('{"schema_version": 1}', encoding="utf-8")
                with patch.object(data_loader, "search_flights", return_value=[]) as search, \
                        patch.object(data_loader.time, "sleep"), self.assertLogs("data_loader", level="WARNING"):
                    self.assertTrue(data_loader.fetch_and_cache_flights().empty)
                self.assertEqual(search.call_count, 2)

    def test_provider_error_does_not_overwrite_cache(self):
        self.write_cache(home_airports=["FKB"])
        path = self.cache_dir / "flight_quotes.json"
        before = path.read_text(encoding="utf-8")
        with patch.object(data_loader, "search_flights", side_effect=flights_api.FlightAPIError("blocked")), \
                patch.object(data_loader.time, "sleep"), self.assertLogs("data_loader", level="WARNING"):
            with self.assertRaises(flights_api.FlightAPIError):
                data_loader.fetch_and_cache_flights()
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_new_fetch_keeps_every_flight_and_correct_timestamps(self):
        flight = dict(from_airport="STR", to_airport="CDG", departure="2026-12-10 10:00",
                      arrival="2026-12-10 14:00", price=100, air_duration_mins=120,
                      stops=1, airlines=["fixture"], co2_kg=417)
        alternatives = [flight, dict(flight, price=130, air_duration_mins=60, stops=0)]
        with patch.object(data_loader, "search_flights", side_effect=[alternatives, []]), \
                patch.object(data_loader.time, "sleep"):
            frame = data_loader.fetch_and_cache_flights()
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["departure_time"], "2026-12-10T10:00:00+01:00")
        self.assertEqual(frame.iloc[0]["arrival_time"], "2026-12-10T14:00:00+01:00")
        saved = json.loads((self.cache_dir / "flight_quotes.json").read_text())
        self.assertEqual(saved["schema_version"], 3)
        self.assertEqual(saved["emissions_source"], "google-flights")
        self.assertEqual(saved["quotes"][0]["co2_kg"], 417)

    def test_optimizer_duration_limit_is_forwarded_and_recorded_without_invalidating_legacy(self):
        with patch.object(config, "MAX_TRANSFER_HOURS", 24.001), \
                patch.object(data_loader, "search_flights", return_value=[]) as search, \
                patch.object(data_loader.time, "sleep"):
            data_loader.fetch_and_cache_flights()
        self.assertTrue(all(call.kwargs["max_duration_minutes"] == 1441 for call in search.call_args_list))
        saved = json.loads((self.cache_dir / "flight_quotes.json").read_text())
        self.assertEqual(saved["query_max_duration_minutes"], 1441)
        self.write_cache()
        output = io.StringIO()
        with patch.object(config, "MAX_TRANSFER_HOURS", 48), \
                patch.object(data_loader, "search_flights") as search, contextlib.redirect_stdout(output):
            data_loader.fetch_and_cache_flights()
        search.assert_not_called()
        self.assertIn("unknown/legacy", output.getvalue())
        self.assertIn("cannot recover unfetched offers", output.getvalue())

    def test_invalid_flight_times_and_endpoint_are_rejected(self):
        for changes in ({"arrival_time": "2026-12-10T09:00+01:00"},
                        {"air_minutes": 300}, {"price": float("nan")},
                        {"co2_kg": float("nan")}, {"co2_kg": -1}, {"co2_kg": 0},
                        {"src_airport": "MUC"}, {"departure_time": "2026-12-10T10:00"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.quote(**changes)

    def test_missing_payload_and_parser_failures_are_explicit(self):
        for text, error in (("captcha", None), (flight_response(), ValueError("schema changed"))):
            with patch.object(flights_api._SESSION, "get", return_value=Mock(text=text)), \
                    patch.object(flights_api, "parse_js", side_effect=error):
                with self.assertRaises(flights_api.FlightAPIError):
                    flights_api.search_flights("2026-12-10", "STR", "CDG")

    def test_transient_timeouts_and_connection_failures_retry_same_query(self):
        response = Mock(text=flight_response())
        with patch.object(flights_api._SESSION, "get", side_effect=[
                flights_api.requests.ReadTimeout("slow response"),
                flights_api.requests.ConnectionError("connection dropped"), response]) as get, \
                patch.object(flights_api.time, "sleep") as sleep, \
                self.assertLogs("api.flights_api", level="WARNING"):
            self.assertEqual(flights_api.search_flights("2026-12-14", "BVA", "HHN"), [])
        self.assertEqual(get.call_count, 3)
        self.assertTrue(all(call == get.call_args_list[0] for call in get.call_args_list))
        self.assertEqual(get.call_args.kwargs["timeout"], (10, 30))
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_exhausted_network_retries_raise_contextual_error(self):
        failure = flights_api.requests.ReadTimeout("slow response")
        with patch.object(flights_api._SESSION, "get", side_effect=failure) as get, \
                patch.object(flights_api.time, "sleep") as sleep, \
                self.assertLogs("api.flights_api", level="WARNING"):
            with self.assertRaisesRegex(flights_api.FlightAPIError, "BVA->HHN on 2026-12-14 after 3 attempts") as caught:
                flights_api.search_flights("2026-12-14", "BVA", "HHN")
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_permanent_http_tls_and_payload_errors_are_not_retried(self):
        for failure in (flights_api.requests.HTTPError("403 Forbidden"),
                        flights_api.requests.exceptions.SSLError("certificate failure"), None):
            response = Mock(text="captcha")
            response.raise_for_status.side_effect = failure if isinstance(
                failure, flights_api.requests.HTTPError) else None
            with self.subTest(failure=failure), \
                    patch.object(flights_api._SESSION, "get", return_value=response,
                                 side_effect=failure if isinstance(
                                     failure, flights_api.requests.exceptions.SSLError) else None) as get, \
                    patch.object(flights_api.time, "sleep") as sleep:
                with self.assertRaises((flights_api.FlightAPIError, flights_api.requests.HTTPError)):
                    flights_api.search_flights("2026-12-14", "BVA", "HHN")
                self.assertEqual(get.call_count, 1)
                sleep.assert_not_called()

    def test_exhausted_retries_preserve_existing_cache(self):
        self.write_cache(home_airports=["FKB"])
        path = self.cache_dir / "flight_quotes.json"
        before = path.read_bytes()
        with patch.object(data_loader, "search_flights", side_effect=flights_api.search_flights), \
                patch.object(flights_api._SESSION, "get", side_effect=flights_api.requests.ReadTimeout("slow")), \
                patch.object(flights_api.time, "sleep"), self.assertLogs(level="WARNING"):
            with self.assertRaises(flights_api.FlightAPIError):
                data_loader.fetch_and_cache_flights()
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(path.with_suffix(".tmp").exists())

    def test_callback_metadata_commas_do_not_become_extra_json_data(self):
        payload = [None, None, None, None, None, None, None, [None, [[], []]]]
        for trailer in (
                "});",
                ",sideChannel: {timings:[1,2], label:'a,b'}});",
                ",sideChannel: {},errorHasStatus: false});"):
            for marker in ("data:", "data : \n ", '"data": '):
                with self.subTest(trailer=trailer, marker=marker):
                    html = ('<script class="ds:1">AF_initDataCallback({key:"ds:1",'
                            + marker + json.dumps(payload) + trailer + "</script>")
                    self.assertEqual(flights_api._parse_flight_response(html), [])

    def test_explicit_provider_error_is_not_cached_as_an_empty_search(self):
        html = flight_response().replace(",other:1", ",sideChannel:{},errorHasStatus: true")
        with patch.object(flights_api._SESSION, "get", return_value=Mock(text=html)):
            with self.assertRaises(flights_api.FlightAPIError) as caught:
                flights_api.search_flights("2026-12-14", "BVA", "HHN")
        self.assertIn("errorHasStatus", str(caught.exception.__cause__))

    def test_multiple_payload_scripts_are_not_silently_concatenated_or_ignored(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            flights_api._parse_flight_response(flight_response() + flight_response())

    def test_malformed_json_and_unexpected_trailing_data_still_fail(self):
        for data in ("[1,2,]", "[1,2] [3,4]", "[1,2]garbage"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                flights_api._parse_flight_response(
                    '<script class="ds:1">callback({data:' + data + ',sideChannel:{}});</script>')

    def test_timezone_localization_preserves_offsets_and_rejects_dst_ambiguity(self):
        self.assertEqual(localize_schedule("2026-12-10 10:00", "London").utcoffset(), timedelta())
        self.assertEqual(localize_schedule("2026-12-10 10:00", "Paris").utcoffset(), timedelta(hours=1))
        for value in ("2026-10-25 02:30", "2026-03-29 02:30"):
            with self.assertRaisesRegex(ValueError, "provider offset required"):
                localize_schedule(value, "Paris")
        self.assertEqual(localize_schedule("2026-10-25T02:30+01:00", "Paris").fold, 1)

    def test_unpriced_offers_do_not_discard_priced_alternatives(self):
        priced = [["flight"], [[None, 156], "booking-token"]]
        unpriced = [["flight"], [[]]]
        html = flight_response(other=[[priced, unpriced]])
        with patch.object(flights_api, "parse_js", return_value=["parsed"]) as parse_js, \
                self.assertLogs("api.flights_api", level="WARNING"):
            result = flights_api._parse_flight_response(html)
        self.assertEqual(result, ["parsed"])
        sanitized = json.loads(parse_js.call_args.args[0].removeprefix("data:").removesuffix(","))
        self.assertEqual(sanitized[3][0], [priced])

    def test_unrelated_schema_failure_is_not_reclassified_as_no_flights(self):
        for section in ({}, [], ["unexpected"], [[["flight", {"price": 156}]]]):
            with self.subTest(section=section), \
                    patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response(other=section))):
                with self.assertRaises(flights_api.FlightAPIError):
                    flights_api.search_flights("2026-12-10", "STR", "LTN")

    def test_null_and_empty_result_sections_are_valid_empty_searches(self):
        for recommended, other in ((None, None), ([None], None), (None, [[]]), ([[]], [None])):
            with self.subTest(recommended=recommended, other=other), \
                    patch.object(flights_api._SESSION, "get",
                                 return_value=Mock(text=flight_response(recommended, other))), \
                    self.assertLogs("api.flights_api", level="INFO"):
                self.assertEqual(flights_api.search_flights("2026-12-10", "STR", "LTN"), [])

    def test_empty_sections_with_invalid_metadata_remain_errors(self):
        for metadata in ([], [None, "invalid"], [None, [[["missing name"]]]]):
            with self.subTest(metadata=metadata), \
                    patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response(metadata=metadata))):
                with self.assertRaises(flights_api.FlightAPIError):
                    flights_api.search_flights("2026-12-10", "STR", "LTN")

    def test_hhn_lgw_empty_response_without_airline_catalogue(self):
        metadata = [
            None,
            [[["ONEWORLD", "Oneworld"], ["SKYTEAM", "SkyTeam"], ["STAR_ALLIANCE", "Star Alliance"]]],
            None, None, None, [1, 0, 0, 0], None, [0],
        ]
        with patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response(metadata=metadata))), \
                self.assertLogs("api.flights_api", level="INFO"):
            self.assertEqual(flights_api.search_flights("2026-12-10", "HHN", "LGW", max_price=None), [])

    def test_optional_catalogues_may_be_absent_null_or_empty(self):
        for catalogues in (None, [], [None], [[]], [[], None], [None, []], [None, None]):
            with self.subTest(catalogues=catalogues):
                self.assertEqual(flights_api._parse_flight_response(
                    flight_response(metadata=[None, catalogues])), [])

    def test_missing_display_catalogue_does_not_discard_priced_flights(self):
        row = [["flight"], [[None, 156]]]
        with patch.object(flights_api, "parse_js", return_value=["priced flight"]) as parse_js:
            result = flights_api._parse_flight_response(
                flight_response(other=[[row]], metadata=[None, [[]]]))
        self.assertEqual(result, ["priced flight"])
        sanitized = json.loads(parse_js.call_args.args[0].removeprefix("data:").removesuffix(","))
        self.assertEqual(sanitized[7][1], [[], []])
        self.assertEqual(sanitized[3][0], [row])

    def test_recommended_and_other_offers_are_both_preserved(self):
        recommended = [["recommended"], [[None, 104]]]
        other = [["other"], [[None, 156]]]
        for other_section in (None, [[other]]):
            with self.subTest(other=other_section), patch.object(flights_api, "parse_js", return_value=[]) as parse_js:
                flights_api._parse_flight_response(flight_response([[recommended]], other_section))
                sanitized = json.loads(parse_js.call_args.args[0].removeprefix("data:").removesuffix(","))
                self.assertEqual(sanitized[3][0], [recommended] + ([other] if other_section else []))

    def test_empty_airport_pair_does_not_abort_or_erase_other_quotes(self):
        valid = dict(from_airport="STR", to_airport="CDG", departure="2026-12-10 10:00",
                     arrival="2026-12-10 14:00", price=100, air_duration_mins=120,
                     stops=1, airlines=["fixture"], co2_kg=417)
        def search(date, src, dest, **kwargs):
            if src == "STR":
                return [valid]
            with patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response())):
                return flights_api.search_flights(date, src, dest, **kwargs)
        with patch.object(data_loader, "search_flights", side_effect=search), \
                patch.object(data_loader.time, "sleep"):
            frame = data_loader.fetch_and_cache_flights()
        self.assertEqual(frame["price"].tolist(), [100])
        self.assertTrue((self.cache_dir / "flight_quotes.json").exists())

    def test_legacy_emissions_cache_is_refetched_not_estimated(self):
        self.write_cache()
        path = self.cache_dir / "flight_quotes.json"
        saved = json.loads(path.read_text())
        saved["schema_version"] = 2
        del saved["quotes"][0]["co2_kg"]
        path.write_text(json.dumps(saved))
        with patch.object(data_loader, "search_flights", return_value=[]) as search, \
                patch.object(data_loader.time, "sleep"), self.assertLogs("data_loader", level="WARNING") as logs:
            self.assertTrue(data_loader.fetch_and_cache_flights().empty)
        self.assertEqual(search.call_count, 2)
        self.assertIn("API emissions", " ".join(logs.output))
        self.assertEqual(json.loads(path.read_text())["schema_version"], 3)

    def test_missing_provider_emissions_are_not_cached_as_estimates(self):
        flight = dict(from_airport="STR", to_airport="CDG")
        with patch.object(data_loader, "search_flights", side_effect=[[flight], []]), \
                patch.object(data_loader.time, "sleep"), self.assertLogs("data_loader", level="WARNING"):
            self.assertTrue(data_loader.fetch_and_cache_flights().empty)
        self.assertEqual(json.loads((self.cache_dir / "flight_quotes.json").read_text())["quotes"], [])

    def provider_flight(self, emission=417000):
        leg = SimpleNamespace(
            from_airport=SimpleNamespace(code="STR"), to_airport=SimpleNamespace(code="CDG"),
            departure=SimpleNamespace(date=(2026, 12, 10), time=(10, 0)),
            arrival=SimpleNamespace(date=(2026, 12, 10), time=(12, 0)), duration=120)
        return SimpleNamespace(flights=[leg], price=100, airlines=["fixture"],
                               carbon=SimpleNamespace(emission=emission, typical_on_route=386000))

    def test_api_converts_actual_emissions_from_grams_not_route_typical(self):
        item = self.provider_flight()
        with patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response())), \
                patch.object(flights_api, "_parse_flight_response", return_value=[item]):
            result = flights_api.search_flights("2026-12-10", "STR", "CDG")
        self.assertEqual(result[0]["co2_kg"], 417)

    def test_api_emissions_survive_cache_roundtrip_and_objective_evaluation(self):
        with patch.object(data_loader, "search_flights", side_effect=flights_api.search_flights), \
                patch.object(data_loader.time, "sleep"), \
                patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response())), \
                patch.object(flights_api, "_parse_flight_response", side_effect=[[self.provider_flight()], []]):
            fresh = data_loader.fetch_and_cache_flights()
        with patch.object(data_loader, "search_flights") as search:
            cached = data_loader.fetch_and_cache_flights()
        search.assert_not_called()
        self.assertEqual(fresh["co2"].tolist(), [417])
        self.assertEqual(cached["co2"].tolist(), [417])
        with patch.object(data_loader, "fetch_and_cache_flights", return_value=cached), \
                patch.object(data_loader, "fetch_and_cache_transit", return_value=data_loader._transit_frame([])):
            conn = data_loader.load_connections()[0]
        from objectives import connection_objectives
        self.assertEqual(connection_objectives(conn)[2], 417)

    def test_api_excludes_missing_emissions_and_retains_valid_alternatives(self):
        for missing in (self.provider_flight(None),
                        SimpleNamespace(**{k: v for k, v in vars(self.provider_flight()).items() if k != "carbon"})):
            with self.subTest(missing=missing), \
                    patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response())), \
                    patch.object(flights_api, "_parse_flight_response",
                                 return_value=[missing, self.provider_flight(83125)]), \
                    self.assertLogs("api.flights_api", level="WARNING"):
                result = flights_api.search_flights("2026-12-10", "STR", "CDG")
            self.assertEqual([f["co2_kg"] for f in result], [83.125])

    def test_api_invalid_emissions_fail_explicitly(self):
        for value in (-1, 0, float("nan"), float("inf"), True, "417000"):
            with self.subTest(value=value), \
                    patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response())), \
                    patch.object(flights_api, "_parse_flight_response", return_value=[self.provider_flight(value)]):
                with self.assertRaisesRegex(flights_api.FlightAPIError, "Invalid provider emissions"):
                    flights_api.search_flights("2026-12-10", "STR", "CDG")

    def test_real_parser_handles_missing_carbon_without_discarding_other_offers(self):
        leg = [None] * 22
        leg[3:7] = ["STR", "Stuttgart", "Paris", "CDG"]
        leg[8], leg[10], leg[11], leg[17] = [10, 0], [12, 0], 120, "fixture aircraft"
        leg[20] = leg[21] = [2026, 12, 10]
        details = ["fixture", ["fixture"], [leg]] + [None] * 20
        details[22] = [None] * 7 + [417000, 386000]
        valid = [details, [[None, 100]]]
        for extras in (None, [], [None] * 8):
            missing = [details[:22] + [extras], [[None, 99]]]
            with self.subTest(extras=extras), \
                    patch.object(flights_api._SESSION, "get",
                                 return_value=Mock(text=flight_response(other=[[missing, valid]]))), \
                    self.assertLogs("api.flights_api", level="WARNING"):
                result = flights_api.search_flights("2026-12-10", "STR", "CDG")
            self.assertEqual([(f["price"], f["co2_kg"]) for f in result], [(100, 417)])

    def test_connecting_flight_emissions_are_not_summed_again_per_leg(self):
        item = self.provider_flight()
        first = item.flights[0]
        first.to_airport = SimpleNamespace(code="FRA")
        second = SimpleNamespace(
            from_airport=SimpleNamespace(code="FRA"), to_airport=SimpleNamespace(code="CDG"),
            departure=SimpleNamespace(date=(2026, 12, 10), time=(13, 0)),
            arrival=SimpleNamespace(date=(2026, 12, 10), time=(14, 0)), duration=60)
        item.flights.append(second)
        with patch.object(flights_api._SESSION, "get", return_value=Mock(text=flight_response())), \
                patch.object(flights_api, "_parse_flight_response", return_value=[item]):
            result = flights_api.search_flights("2026-12-10", "STR", "CDG")
        self.assertEqual(result[0]["co2_kg"], 417)
        self.assertEqual(result[0]["stops"], 1)
        self.assertEqual(result[0]["layover_duration_mins"], 60)


if __name__ == "__main__":
    unittest.main()
