import base64
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

for _sub in ("src", "src/helpers", "src/optimizer"): sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))

import config
import data_loader
from api import hotels_api


def offer(city="Paris", check_in_date="2026-12-10", check_out_date="2026-12-11", **changes):
    result = dict(
        city=city, check_in_date=check_in_date, check_out_date=check_out_date,
        name="Example Hotel", property_token="property-1", price_per_night=80.0,
        currency="EUR", stars=3, rating=4.2, reviews_count=100, category="gcid:hotel",
        latitude=48.8566, longitude=2.3522, address="Example Street", hex_id=None,
    )
    result.update(changes)
    result.setdefault("base_stay_price", result["price_per_night"])
    result.setdefault("co2_kg_per_night", {1: 9.0, 2: 11.5, 3: 15.5, 4: 24.0, 5: 42.0}.get(result["stars"], 15.5))
    result.setdefault("total_co2_kg", result["co2_kg_per_night"])
    result.setdefault("check_in_time", "3:00 PM")
    result.setdefault("check_out_time", "11:00 AM")
    result.setdefault("taxes_and_fees", 0.0)
    result.setdefault("total_price", result["base_stay_price"] + result["taxes_and_fees"])
    return result


def hotel_html(offers, *, adults=1, search_dates=None, include_context=True, include_quote_dates=True):
    cards = []
    for item in offers:
        hotel = [None] * 21
        details = [None] * 32
        details[0] = [item["latitude"], item["longitude"]]
        details[1] = [[[item["address"]]]]
        details[31] = [[[item["category"]]]]
        hotel[1], hotel[2], hotel[3] = item["name"], details, [None, item["stars"]]
        start, end = (datetime.fromisoformat(item[key]) for key in ("check_in_date", "check_out_date"))
        dates = [[start.year, start.month, start.day], [end.year, end.month, end.day], (end - start).days]
        prices = [None] * 45
        prices[1] = [None, None, item["price_per_night"]]
        prices[15] = item["currency"]
        prices[44] = [item["base_stay_price"], item["taxes_and_fees"], 0, item["total_price"]]
        hotel[6] = [None, [None, None, None, item["currency"], dates if include_quote_dates else None], prices]
        hotel[7], hotel[9], hotel[20] = [[item["rating"], item["reviews_count"]]], item["hex_id"], item["property_token"]
        cards.append([34, {"397419284": [hotel]}])
    payload = [[[[None, cards]]]]
    if include_context:
        first = offers[0] if offers else offer()
        start, end = (datetime.fromisoformat(value) for value in (
            search_dates or (first["check_in_date"], first["check_out_date"])))
        dates = [[start.year, start.month, start.day], [end.year, end.month, end.day], (end - start).days]
        context = [1, [[[3]], 1], [None, [None, dates, None, None, None, [adults]]],
                   None, [[None] * 6 + [first["currency"]]]]
        payload.append([context])
    return "<script>AF_initDataCallback({key: 'ds:0', data:" + json.dumps(payload) + "});</script>"


def protobuf_fields(data):
    def read_varint():
        nonlocal offset
        value = shift = 0
        while True:
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    offset = 0
    fields = {}
    while offset < len(data):
        tag = read_varint()
        if tag & 7 == 0:
            value = read_varint()
        elif tag & 7 == 2:
            length = read_varint()
            value = data[offset:offset + length]
            offset += length
        else:
            raise AssertionError(f"Unexpected protobuf wire type {tag & 7}")
        fields[tag >> 3] = value
    return fields


class HotelCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.cache_dir = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.multiple(
            data_loader, CACHE_DIR=self.cache_dir, CITIES={0: "Stuttgart", 1: "Paris"},
            TRAVEL_RANGE=["2026-12-10", "2026-12-13"],
        ))
        self.stack.enter_context(patch.multiple(
            config, OPTIMIZE_HOTELS=True, HOTEL_MIN_STARS=3, HOTEL_MIN_GUEST_RATING=3.5,
            HOTEL_MIN_REVIEWS_COUNT=30, HOTEL_MAX_PRICE_PER_NIGHT=150.0,
            HOTEL_INCLUDE_UNSTARRED=False, NUM_TRAVELLERS=1,
            HOTEL_REQUEST_DELAY_SECONDS=0.0,
        ))
        self.search = self.stack.enter_context(patch.object(
            data_loader, "search_hotels",
            side_effect=lambda city, check_in_date, check_out_date, **_: [
                offer(city, check_in_date, check_out_date)],
        ))

    def test_reports_fetching_or_cache_loading(self):
        fetched = io.StringIO()
        with contextlib.redirect_stdout(fetched):
            data_loader.fetch_and_cache_hotels()
        self.assertEqual(fetched.getvalue(), "Fetching hotel quotes from API...\n")

        self.search.reset_mock()
        cached = io.StringIO()
        with contextlib.redirect_stdout(cached):
            data_loader.fetch_and_cache_hotels()
        self.search.assert_not_called()
        self.assertEqual(cached.getvalue(),
                         f"Loaded hotel quotes from cache ({self.cache_dir / 'hotel_quotes.csv'})\n")

        refreshed = io.StringIO()
        with patch.object(config, "NUM_TRAVELLERS", 2), \
                self.assertLogs("data_loader", level="WARNING"), \
                contextlib.redirect_stdout(refreshed):
            data_loader.fetch_and_cache_hotels()
        self.assertEqual(refreshed.getvalue(), "Fetching hotel quotes from API...\n")
        self.assertEqual(self.search.call_count, 3)

    def test_csv_round_trip_preserves_every_night_and_alternative(self):
        self.search.side_effect = lambda city, check_in_date, check_out_date, **_: [
            offer(city, check_in_date, check_out_date),
            offer(city, check_in_date, check_out_date, property_token="property-2", price_per_night=95),
        ]
        original = data_loader.fetch_and_cache_hotels()
        self.assertEqual(len(original), 6)
        self.assertEqual(self.search.call_count, 3)
        for day, call in zip((10, 11, 12), self.search.call_args_list):
            self.assertEqual(call.kwargs["check_in_date"], f"2026-12-{day}")
            self.assertEqual(call.kwargs["check_out_date"], f"2026-12-{day + 1}")
            self.assertEqual(call.kwargs["adults"], 1)
            self.assertEqual(call.kwargs["min_reviews"], 30)
            self.assertEqual(call.kwargs["currency"], "EUR")
            self.assertNotIn("max_total_price", call.kwargs)
            self.assertIn("include_unstarred", call.kwargs)
            self.assertNotIn("eco_certified_only", call.kwargs)
        self.assertTrue((self.cache_dir / "hotel_quotes.csv").exists())
        self.search.reset_mock()
        cached = data_loader.fetch_and_cache_hotels()
        self.search.assert_not_called()
        self.assertEqual(original.to_dict("records"), cached.to_dict("records"))
        self.assertEqual(cached.iloc[0]["category"], "gcid:hotel")
        self.assertNotIn("eco_certified", cached.columns)

    def test_disabled_planning_does_not_fetch_or_touch_rng(self):
        before = random.getstate()
        with patch.object(config, "OPTIMIZE_HOTELS", False):
            self.assertEqual(data_loader.load_hotels(), [])
        self.search.assert_not_called()
        self.assertEqual(list(self.cache_dir.iterdir()), [])
        self.assertEqual(random.getstate(), before)

    def test_empty_search_grid_is_cached(self):
        self.search.side_effect = None
        self.search.return_value = []
        self.assertTrue(data_loader.fetch_and_cache_hotels().empty)
        self.search.reset_mock()
        self.assertTrue(data_loader.fetch_and_cache_hotels().empty)
        self.search.assert_not_called()

    def test_every_search_setting_invalidates_cache(self):
        for key, value in (
            ("HOTEL_MIN_STARS", 2), ("HOTEL_MIN_GUEST_RATING", 3),
            ("HOTEL_MIN_REVIEWS_COUNT", 10), ("HOTEL_MAX_PRICE_PER_NIGHT", None),
            ("HOTEL_INCLUDE_UNSTARRED", True), ("NUM_TRAVELLERS", 2),
        ):
            with self.subTest(key=key):
                self.search.side_effect = lambda city, check_in_date, check_out_date, **_: [
                    offer(city, check_in_date, check_out_date)]
                data_loader.fetch_and_cache_hotels()
                self.search.reset_mock()
                with patch.object(config, key, value), self.assertLogs("data_loader", level="WARNING"):
                    data_loader.fetch_and_cache_hotels()
                self.assertEqual(self.search.call_count, 3)

    def test_changed_cities_and_travel_range_refetch(self):
        data_loader.fetch_and_cache_hotels()
        self.search.reset_mock()
        with patch.multiple(data_loader, CITIES={0: "Stuttgart", 1: "Rome"},
                            TRAVEL_RANGE=["2026-12-11", "2026-12-13"]), \
                self.assertLogs("data_loader", level="WARNING"):
            result = data_loader.fetch_and_cache_hotels()
        self.assertEqual(self.search.call_count, 2)
        self.assertEqual(set(result["city"]), {"Rome"})

    def test_cache_and_quotes_do_not_expire(self):
        real_datetime = datetime
        old = datetime.now(timezone.utc) - timedelta(days=3650)
        with patch.object(data_loader, "datetime", wraps=real_datetime) as clock:
            clock.now.return_value = old
            original = data_loader.fetch_and_cache_hotels()
        self.search.reset_mock()
        cached = data_loader.fetch_and_cache_hotels()
        self.search.assert_not_called()
        self.assertEqual(original.to_dict("records"), cached.to_dict("records"))
        self.assertEqual(datetime.fromisoformat(cached.iloc[0]["quoted_at"]), old)

    def test_empty_cache_does_not_expire(self):
        self.search.side_effect = None
        self.search.return_value = []
        data_loader.fetch_and_cache_hotels()
        path = self.cache_dir / "hotel_quotes.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["fetched_at"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.search.reset_mock()
        self.assertTrue(data_loader.fetch_and_cache_hotels().empty)
        self.search.assert_not_called()

    def test_corrupt_and_incomplete_caches_refetch(self):
        csv_path = self.cache_dir / "hotel_quotes.csv"
        manifest_path = csv_path.with_suffix(".json")
        for corruption in ("checksum", "schema", "missing"):
            with self.subTest(corruption=corruption):
                data_loader.fetch_and_cache_hotels()
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if corruption == "schema":
                    manifest["schema_version"] = 1
                elif corruption == "checksum":
                    csv_path.write_text("broken", encoding="utf-8")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                if corruption == "missing":
                    manifest_path.unlink()
                self.search.reset_mock()
                with self.assertLogs("data_loader", level="WARNING"):
                    data_loader.fetch_and_cache_hotels()
                self.assertEqual(self.search.call_count, 3)

    def test_natural_language_cache_is_refetched_even_when_empty(self):
        self.search.side_effect = None
        self.search.return_value = []
        data_loader.fetch_and_cache_hotels()
        path = self.cache_dir / "hotel_quotes.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 3
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.search.reset_mock()
        with self.assertLogs("data_loader", level="WARNING"):
            data_loader.fetch_and_cache_hotels()
        self.assertEqual(self.search.call_count, 3)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 4)

    def test_wrong_date_html_does_not_publish_a_cache(self):
        self.search.side_effect = hotels_api.search_hotels
        html = hotel_html([offer(check_in_date="2026-11-08", check_out_date="2026-11-09")])
        with patch.dict(hotels_api._RAW_QUERY_CACHE, clear=True), \
                patch.object(hotels_api, "_MIN_REQUEST_INTERVAL_SEC", 0), \
                patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)):
            with self.assertRaisesRegex(hotels_api.HotelAPIError, "search dates"):
                data_loader.fetch_and_cache_hotels()
            self.assertFalse(hotels_api._RAW_QUERY_CACHE)
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_provider_failure_keeps_previous_cache_and_no_partial_files(self):
        data_loader.fetch_and_cache_hotels()
        before = {p.name: p.read_bytes() for p in self.cache_dir.iterdir()}
        self.search.side_effect = [[], hotels_api.HotelAPIError("blocked")]
        with patch.object(config, "NUM_TRAVELLERS", 2), self.assertLogs("data_loader", level="WARNING"):
            with self.assertRaises(hotels_api.HotelAPIError):
                data_loader.fetch_and_cache_hotels()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.cache_dir.iterdir()})

    def test_wrong_city_dates_currency_and_invalid_numbers_are_rejected(self):
        for changes in (
            {"city": "Rome"}, {"check_in_date": "2026-12-09"},
            {"check_out_date": "2026-12-12"}, {"currency": "USD"},
            {"price_per_night": float("nan")}, {"price_per_night": 0},
            {"stars": 6}, {"rating": 6}, {"property_token": " "},
            {"total_price": float("inf")}, {"taxes_and_fees": -1},
            {"total_price": 100}, {"base_stay_price": -1},
            {"category": ""}, {"price_per_night": 151},
        ):
            with self.subTest(changes=changes):
                self.search.side_effect = None
                self.search.return_value = [offer(**changes)]
                with self.assertRaises(hotels_api.HotelAPIError):
                    data_loader.fetch_and_cache_hotels()
                self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_config_validation_happens_before_network(self):
        for key, value in (
            ("NUM_TRAVELLERS", 0), ("NUM_TRAVELLERS", 1.5), ("HOTEL_MIN_GUEST_RATING", 7),
            ("HOTEL_MIN_STARS", 6), ("HOTEL_MAX_PRICE_PER_NIGHT", -1),
            ("HOTEL_REQUEST_DELAY_SECONDS", -1),
        ):
            with self.subTest(key=key), patch.object(config, key, value):
                with self.assertRaises(ValueError):
                    data_loader.fetch_and_cache_hotels()
        self.search.assert_not_called()

    def test_loader_preserves_identity_gaps_and_configured_estimates(self):
        self.search.side_effect = [
            [offer(), offer(property_token="property-2", stars=5)],
            [],
            [offer(check_in_date="2026-12-12", check_out_date="2026-12-13", price_per_night=95)],
        ]
        with self.assertLogs("data_loader", level="WARNING"):
            hotels = data_loader.load_hotels()
        self.assertEqual(len(hotels), 2)
        mid, premium = hotels
        self.assertEqual(mid.daily_rates, {0: 80, 2: 95})
        self.assertEqual(premium.daily_rates, {0: 80})
        self.assertEqual((mid.tier, premium.tier), (2, 3))
        self.assertEqual(mid.co2_per_night, 15.5)
        self.assertEqual(premium.co2_per_night, 42.0)
        self.assertEqual((mid.check_in_hour, mid.check_out_hour), (15.0, 11.0))
        self.assertFalse(any(h.synthetic for h in hotels))
        self.search.reset_mock()
        with patch.dict(config.HOTEL_DISCOMFORT_PER_NIGHT_BY_STARS, {3: 7.0}), \
                self.assertLogs("data_loader", level="WARNING"):
            hotels = data_loader.load_hotels()
        self.search.assert_not_called()
        self.assertEqual((hotels[0].co2_per_night, hotels[0].discomfort), (15.5, 7))

    def test_all_star_classes_map_to_optimizer_tiers_without_losing_alternatives(self):
        self.search.side_effect = lambda city, check_in_date, check_out_date, **_: [
            offer(city, check_in_date, check_out_date, stars=stars, property_token=f"property-{stars}")
            for stars in range(1, 6)
        ]
        with patch.object(config, "HOTEL_MIN_STARS", 1), self.assertLogs("data_loader", level="WARNING"):
            hotels = data_loader.load_hotels()
        self.assertEqual([hotel.tier for hotel in hotels], [1, 1, 2, 3, 3])
        self.assertTrue(all(hotel.daily_rates == {0: 80, 1: 80, 2: 80} for hotel in hotels))

    def test_unclassified_and_hostels_have_explicit_local_estimates(self):
        self.search.side_effect = lambda city, check_in_date, check_out_date, **_: [
            offer(city, check_in_date, check_out_date, stars=0, co2_kg_per_night=15.5),
            offer(city, check_in_date, check_out_date, stars=0, property_token="property-2",
                  category="gcid:hostel", co2_kg_per_night=7.5),
        ]
        with patch.object(config, "HOTEL_INCLUDE_UNSTARRED", True), self.assertLogs("data_loader", level="WARNING"):
            unclassified, hostel = data_loader.load_hotels()
        self.assertEqual((unclassified.tier, hostel.tier), (2, 1))
        self.assertEqual(unclassified.co2_per_night, 15.5)
        self.assertEqual(unclassified.discomfort, config.HOTEL_DISCOMFORT_PER_NIGHT_BY_STARS[0])
        self.assertEqual(hostel.co2_per_night, 7.5)
        self.assertEqual(hostel.discomfort, config.HOTEL_HOSTEL_DISCOMFORT_PER_NIGHT)

    def test_one_night_totals_include_fees_and_stable_hex_identity(self):
        self.search.side_effect = lambda city, check_in_date, check_out_date, **_: [
            offer(city, check_in_date, check_out_date, property_token=check_in_date,
                  hex_id="0x123:0x456", taxes_and_fees=20),
            offer(city, check_in_date, check_out_date, property_token=check_in_date + "-second",
                  hex_id="0x123:0x456", price_per_night=70, taxes_and_fees=40),
        ]
        with self.assertLogs("data_loader", level="WARNING"):
            hotels = data_loader.load_hotels()
        self.assertEqual(len(hotels), 1)
        self.assertEqual(hotels[0].property_token, "0x123:0x456")
        self.assertEqual(hotels[0].daily_rates, {0: 100, 1: 100, 2: 100})
        frame = data_loader.fetch_and_cache_hotels()
        self.assertEqual(frame.iloc[0]["base_stay_price"], 80)
        self.assertEqual(frame.iloc[0]["taxes_and_fees"], 20)
        self.assertEqual(frame.iloc[0]["total_price"], 100)

    def test_real_api_contract_flows_from_html_through_csv_to_hotels(self):
        self.search.side_effect = hotels_api.search_hotels
        responses = [
            Mock(text=hotel_html([offer(check_in_date=f"2026-12-{day}", check_out_date=f"2026-12-{day + 1}",
                                        taxes_and_fees=12)]))
            for day in (10, 11, 12)
        ]
        with patch.dict(hotels_api._RAW_QUERY_CACHE, clear=True), \
                patch.object(hotels_api, "_MIN_REQUEST_INTERVAL_SEC", 0), \
                patch.object(hotels_api._SESSION, "get", side_effect=responses) as fetch:
            hotels = data_loader.load_hotels()
            self.assertEqual(fetch.call_count, 3)
            cached = data_loader.load_hotels()
            self.assertEqual(fetch.call_count, 3)
        self.assertEqual(hotels, cached)
        self.assertEqual(hotels[0].daily_rates, {0: 92, 1: 92, 2: 92})


class HotelAPITests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(hotels_api._RAW_QUERY_CACHE, clear=True))
        self.stack.enter_context(patch.object(hotels_api, "_MIN_REQUEST_INTERVAL_SEC", 0))

    def search(self, **kwargs):
        return hotels_api.search_hotels("Paris", "2026-12-10", "2026-12-11", **kwargs)

    def test_dated_search_includes_occupancy_and_never_falls_back(self):
        with patch.object(hotels_api._SESSION, "get",
                          return_value=Mock(text=hotel_html([offer()], adults=2))) as fetch:
            self.assertEqual(len(self.search(adults=2)), 1)
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args[0], "https://www.google.com/travel/hotels")
        params = fetch.call_args.kwargs["params"]
        self.assertEqual(params["q"], "Paris")
        state = protobuf_fields(base64.urlsafe_b64decode(params["ts"] + "=" * (-len(params["ts"]) % 4)))
        stay = protobuf_fields(protobuf_fields(state[3])[2])
        dates = protobuf_fields(stay[2])
        self.assertEqual(protobuf_fields(dates[1]), {1: 2026, 2: 12, 3: 10})
        self.assertEqual(protobuf_fields(dates[2]), {1: 2026, 2: 12, 3: 11})
        self.assertEqual(dates[3], 1)
        self.assertEqual(protobuf_fields(stay[6]), {1: 2})
        self.assertEqual(protobuf_fields(protobuf_fields(state[5])[1])[7], b"EUR")
        self.assertNotIn("ap", params)
        self.assertNotIn("rp", params)
        with patch.object(hotels_api._SESSION, "get", return_value=Mock(text=hotel_html([]))) as fetch:
            self.assertEqual(self.search(), [])
            self.assertEqual(self.search(), [])
        fetch.assert_called_once()

    def test_search_state_handles_year_boundary_and_large_varints(self):
        state = hotels_api._hotel_search_state(
            datetime(2027, 12, 31), datetime(2028, 1, 2), 130, "USD")
        root = protobuf_fields(base64.urlsafe_b64decode(state + "=" * (-len(state) % 4)))
        stay = protobuf_fields(protobuf_fields(root[3])[2])
        dates = protobuf_fields(stay[2])
        self.assertEqual(protobuf_fields(dates[1]), {1: 2027, 2: 12, 3: 31})
        self.assertEqual(protobuf_fields(dates[2]), {1: 2028, 2: 1, 3: 2})
        self.assertEqual(dates[3], 2)
        self.assertEqual(protobuf_fields(stay[6])[1], 130)
        self.assertEqual(protobuf_fields(protobuf_fields(root[5])[1])[7], b"USD")

    def test_missing_context_or_wrong_occupancy_is_not_cached(self):
        for html in (hotel_html([], include_context=False), hotel_html([], adults=2)):
            with self.subTest(html=html), \
                    patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)):
                with self.assertRaises(hotels_api.HotelAPIError):
                    self.search()
                self.assertFalse(hotels_api._RAW_QUERY_CACHE)

    def test_wrong_dates_in_empty_search_context_are_not_cached(self):
        html = hotel_html([], search_dates=("2026-11-08", "2026-11-09"))
        with patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)):
            with self.assertRaisesRegex(hotels_api.HotelAPIError, "search dates"):
                self.search()
        self.assertFalse(hotels_api._RAW_QUERY_CACHE)

    def test_mixed_date_cards_preserve_valid_alternatives(self):
        html = hotel_html([offer(), offer(check_in_date="2026-11-08", check_out_date="2026-11-09")])
        with patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)), \
                self.assertLogs("api.hotels_api", level="WARNING"):
            self.assertEqual(len(self.search()), 1)

    def test_unusable_cards_are_not_a_valid_empty_search(self):
        cases = [
            hotel_html([offer(check_in_date="2026-11-08", check_out_date="2026-11-09")],
                       search_dates=("2026-12-10", "2026-12-11")),
            hotel_html([offer()], include_quote_dates=False),
            hotel_html([offer(price_per_night=None, base_stay_price=80)]),
        ]
        for html in cases:
            with self.subTest(html=html), \
                    patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)), \
                    self.assertLogs("api.hotels_api", level="WARNING"):
                with self.assertRaisesRegex(hotels_api.HotelAPIError, "not a verified empty search"):
                    self.search()
                self.assertFalse(hotels_api._RAW_QUERY_CACHE)

    def test_structured_query_cache_separates_dates_occupancy_currency_and_city(self):
        requests_to_make = [
            ("Paris", "2026-12-10", "2026-12-11", 1, "EUR"),
            ("Paris", "2026-12-11", "2026-12-12", 1, "EUR"),
            ("Paris", "2026-12-10", "2026-12-11", 2, "EUR"),
            ("Paris", "2026-12-10", "2026-12-11", 1, "USD"),
            ("London", "2026-12-10", "2026-12-11", 1, "EUR"),
        ]
        for city, start, end, adults, currency in requests_to_make:
            html = hotel_html([offer(city, start, end, currency=currency)], adults=adults)
            with patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)) as fetch:
                for _ in range(2):
                    self.assertEqual(len(hotels_api.search_hotels(
                        city, start, end, adults=adults, currency=currency)), 1)
                fetch.assert_called_once()
        self.assertEqual(len(hotels_api._RAW_QUERY_CACHE), len(requests_to_make))

    def test_missing_quality_does_not_bypass_filters(self):
        for changes, filters in (
            ({"stars": 0}, {"min_stars": 3}),
            ({"rating": 0}, {"min_rating": 3.5}),
            ({"reviews_count": 0}, {"min_reviews": 30}),
            ({"price_per_night": 151}, {"max_price_per_night": 150}),
        ):
            with self.subTest(changes=changes), \
                    patch.dict(hotels_api._RAW_QUERY_CACHE, clear=True), \
                    patch.object(hotels_api._SESSION, "get", return_value=Mock(text=hotel_html([offer(**changes)]))):
                self.assertEqual(self.search(**filters), [])

    def test_unstarred_rules_and_category_estimates(self):
        for category in ("gcid:hotel", "gcid:hostel"):
            with self.subTest(category=category), patch.dict(hotels_api._RAW_QUERY_CACHE, clear=True), \
                    patch.object(hotels_api._SESSION, "get",
                                 return_value=Mock(text=hotel_html([offer(stars=0, category=category)]))) as fetch:
                self.assertEqual(self.search(min_stars=3), [])
                allowed = self.search(min_stars=3, include_unstarred=True)
                self.assertEqual(len(allowed), 1)
                self.assertEqual(allowed[0]["co2_kg_per_night"], 7.5 if category == "gcid:hostel" else 15.5)
                self.assertEqual(len(self.search(min_stars=1)), 1)
                self.assertNotIn("eco_certified", allowed[0])
                fetch.assert_called_once()

    def test_different_currency_is_not_silently_labeled_eur(self):
        with patch.object(hotels_api._SESSION, "get", return_value=Mock(text=hotel_html([offer(currency="USD")]))):
            with self.assertRaisesRegex(hotels_api.HotelAPIError, "currency"):
                self.search()

    def test_price_cap_uses_total_including_taxes_and_is_inclusive(self):
        with patch.object(hotels_api._SESSION, "get",
                          return_value=Mock(text=hotel_html([offer(price_per_night=80, taxes_and_fees=30)]))) as fetch:
            self.assertEqual(self.search(max_price_per_night=100), [])
            allowed = self.search(max_price_per_night=110)
            self.assertEqual(len(allowed), 1)
            self.assertEqual(allowed[0]["total_price"], 110)
            self.assertEqual(self.search(max_total_price=109), [])
            fetch.assert_called_once()

    def test_provider_date_mismatch_is_not_cached(self):
        html = hotel_html([offer(check_in_date="2026-12-11", check_out_date="2026-12-12")])
        with patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)):
            with self.assertRaises(hotels_api.HotelAPIError):
                self.search()
        self.assertFalse(hotels_api._RAW_QUERY_CACHE)

    def test_invalid_tax_breakdowns_are_not_cached_as_valid_prices(self):
        for changes in ({"taxes_and_fees": -1}, {"total_price": float("nan")},
                        {"total_price": 0}, {"total_price": 100}):
            with self.subTest(changes=changes), \
                    patch.object(hotels_api._SESSION, "get",
                                 return_value=Mock(text=hotel_html([offer(**changes)]))):
                with self.assertRaises(hotels_api.HotelAPIError):
                    self.search()
                self.assertFalse(hotels_api._RAW_QUERY_CACHE)

    def test_unrecognized_payload_and_http_errors_are_explicit(self):
        for html in ("captcha", "<script>AF_initDataCallback({data:[1,2,3]});</script>"):
            with self.subTest(html=html), \
                    patch.object(hotels_api._SESSION, "get", return_value=Mock(text=html)):
                with self.assertRaises(hotels_api.HotelAPIError):
                    self.search()
        with patch.object(hotels_api._SESSION, "get", side_effect=requests.Timeout("timeout")):
            with self.assertRaises(hotels_api.HotelAPIError):
                self.search()
        self.assertFalse(hotels_api._RAW_QUERY_CACHE)


if __name__ == "__main__":
    unittest.main()
