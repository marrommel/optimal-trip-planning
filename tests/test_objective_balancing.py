import contextlib
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import io
import random
import unittest
from unittest.mock import patch

from test_optimizer import config, connection, data_loader, genome, main, objectives
from models import Hotel, Phenotype
from progress import ParetoArchive, QualityMonitor
from timing import clock_window_overlap, local_time, sleep_overlap


class ObjectiveBalancingTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, NUM_DESTINATIONS=2, TOTAL_DAYS=5, OPTIMIZE_HOTELS=True,
            TRAVEL_RANGE=["2026-12-10", "2026-12-23"], MAX_TRANSFER_HOURS=24,
            BALANCED_SOLUTION_WEIGHTS=(0.5, 0.25, 0.25), ALLOW_SYNTHETIC_HOTELS=False,
            TRAVEL_DISCOMFORT_PER_HOUR={"bus": 1}, TRANSFER_DISCOMFORT=0,
            SLEEP_DISRUPTION_PER_HOUR=0, DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR={"bus": 0},
            NIGHT_TRAVEL_DISCOMFORT_PER_HOUR={"bus": 0}, LONG_JOURNEY_DISCOMFORT_PER_HOUR=0))
        self.stack.enter_context(patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True))
        self.connections = [
            connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T10:00+00:00", cost=10, co2=1),
            connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00", cost=30, co2=3),
            connection(1, 0, "2026-12-12T09:00+00:00", "2026-12-12T16:00+01:00", cost=100, co2=20),
        ]
        self.install(self.connections)
        self.hotels = [
            Hotel(1, 1, {0: 10, 1: 30}, 2, 3),
            Hotel(1, 2, {0: 40}, 4, 5),
            Hotel(2, 2, {0: 70}, 7, 8),
        ]
        self.stack.enter_context(patch.object(objectives, "HOTELS_DB", self.hotels))

    def install(self, connections):
        data_loader.CONNECTIONS_BY_ROUTE.clear()
        for conn in connections:
            data_loader.CONNECTIONS_BY_ROUTE.setdefault((conn.day, conn.src, conn.dest), []).append(conn)

    def candidate(self, values, identifier):
        ind = genome()
        ind.objectives = values
        ind.phenotype = Phenotype([0, 1, 0], [2], [
            replace(self.connections[0], journey_id=identifier), self.connections[-1]], [])
        return ind

    def test_scales_match_hand_calculated_route_city_and_trip_medians(self):
        # Three legs * (60, 4.5, 11), plus four nights * (50, 5, 6).
        balance = objectives.ObjectiveBalance.from_data()
        self.assertEqual(balance.scales, (380, 33.5, 57))
        self.assertEqual(balance.weights, (0.5, 0.25, 0.25))
        self.assertAlmostEqual(balance.score(balance.scales), 1)

    def test_quote_counts_do_not_reweight_routes_or_cities(self):
        before = objectives.ObjectiveBalance.from_data()
        self.install(self.connections + self.connections[:2] * 100)
        self.hotels.extend(self.hotels[:2] * 100)
        self.assertEqual(objectives.ObjectiveBalance.from_data(), before)

    def test_trip_length_hotel_toggle_and_loaded_party_totals(self):
        with patch.object(config, "OPTIMIZE_HOTELS", False):
            self.assertEqual(objectives.ObjectiveBalance.from_data().scales, (180, 13.5, 33))
        with patch.multiple(config, NUM_DESTINATIONS=3, TOTAL_DAYS=8):
            self.assertEqual(objectives.ObjectiveBalance.from_data().scales, (590, 53, 86))
        with patch.object(config, "NUM_TRAVELLERS", 4):
            self.assertEqual(objectives.ObjectiveBalance.from_data().scales, (380, 33.5, 57))

    def test_currency_and_emissions_unit_changes_preserve_scores(self):
        original = objectives.ObjectiveBalance.from_data()
        for cost_factor, carbon_factor in ((100, 1000), (0.001, 0.00001)):
            with self.subTest(factors=(cost_factor, carbon_factor)):
                self.install([replace(c, cost=c.cost * cost_factor, co2=c.co2 * carbon_factor)
                              for c in self.connections])
                scaled_hotels = [replace(h, daily_rates={d: v * cost_factor for d, v in h.daily_rates.items()},
                                         co2_per_night=h.co2_per_night * carbon_factor) for h in self.hotels]
                with patch.object(objectives, "HOTELS_DB", scaled_hotels):
                    scaled = objectives.ObjectiveBalance.from_data()
                for actual, expected in zip(scaled.scales, (
                        original.scales[0] * cost_factor, original.scales[1], original.scales[2] * carbon_factor)):
                    self.assertAlmostEqual(actual, expected)
                for values in ((200, 40, 50), (400, 20, 30)):
                    self.assertAlmostEqual(original.score(values),
                                           scaled.score((values[0] * cost_factor, values[1], values[2] * carbon_factor)))

    def test_ineligible_and_unavailable_quotes_do_not_affect_scales(self):
        before = objectives.ObjectiveBalance.from_data()
        too_long = replace(self.connections[0], arrival_time=self.connections[0].arrival_time + timedelta(days=2),
                           cost=999999, co2=999999)
        self.install(self.connections + [too_long] * 100)
        self.hotels.extend([Hotel(1, 1, {}, 999, 999), Hotel(0, 1, {0: 99999}, 999, 999),
                            Hotel(1, 1, {0: 99999}, 999, 999, synthetic=True)])
        self.assertEqual(objectives.ObjectiveBalance.from_data(), before)

    def test_zero_axes_and_sparse_positive_axes_have_valid_data_based_scales(self):
        with patch.multiple(config, OPTIMIZE_HOTELS=False, TRAVEL_DISCOMFORT_PER_HOUR={"bus": 0}):
            self.install([replace(c, cost=0, co2=0) for c in self.connections])
            balance = objectives.ObjectiveBalance.from_data()
            self.assertEqual(balance.scales, (1, 1, 1))
            self.assertEqual(balance.score((0, 0, 0)), 0)
            for value in (0.01, 0.00001):
                self.install([replace(self.connections[0], co2=c) for c in (0, 0, value)])
                self.assertAlmostEqual(objectives.ObjectiveBalance.from_data().scales[2], 3 * value)

    def test_missing_transport_and_invalid_objectives_fail_explicitly(self):
        self.install([])
        with self.assertRaisesRegex(ValueError, "no duration-eligible transport quotes"):
            objectives.ObjectiveBalance.from_data()
        for value in (-1, float("nan"), float("inf")):
            self.install([replace(self.connections[0], cost=value)])
            with self.subTest(transport=value), self.assertRaisesRegex(ValueError, "invalid transport"):
                objectives.ObjectiveBalance.from_data()
            self.install(self.connections)
            with patch.object(objectives, "HOTELS_DB", [replace(self.hotels[0], daily_rates={0: value})]):
                with self.subTest(hotel=value), self.assertRaisesRegex(ValueError, "invalid hotel"):
                    objectives.ObjectiveBalance.from_data()

    def test_invalid_scoring_inputs_and_weights_fail_explicitly(self):
        for weights in ((0, 0, 0), (-1, 1, 1), (1, 1), (1, 1, float("inf"))):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "BALANCED_SOLUTION_WEIGHTS"):
                objectives.ObjectiveBalance((1, 1, 1), weights)
        balance = objectives.ObjectiveBalance.from_data()
        for values in ((1, 1), (-1, 1, 1), (1, 1, float("nan"))):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "three finite non-negative"):
                balance.score(values)

    def test_reference_is_deterministic_immutable_and_does_not_use_randomness(self):
        before = random.getstate()
        balance = objectives.ObjectiveBalance.from_data()
        self.install(list(reversed(self.connections)))
        self.hotels.reverse()
        self.assertEqual(objectives.ObjectiveBalance.from_data(), balance)
        self.assertEqual(random.getstate(), before)
        with self.assertRaises(FrozenInstanceError):
            balance.scales = (1, 1, 1)

    def test_monitor_and_final_selection_share_frozen_scoring(self):
        balance = objectives.ObjectiveBalance.from_data()
        archive = ParetoArchive()
        a = self.candidate((200, 40, 50), "a")
        b = self.candidate((400, 20, 30), "b")
        archive.update([a, b])
        monitor = QualityMonitor(balance)
        snapshot = monitor.observe(archive)
        selected = main.select_best_options(archive.front, balance)["Balanced Optimum"]
        self.assertEqual(snapshot.compromise, balance.score(selected.objectives))
        with patch.object(config, "BALANCED_SOLUTION_WEIGHTS", (0, 0, 1)):
            self.install([replace(c, cost=c.cost * 100) for c in self.connections])
            self.assertEqual(monitor.observe(archive).compromise, snapshot.compromise)
            self.assertIs(main.select_best_options(archive.front, balance)["Balanced Optimum"], selected)

    def test_main_derives_once_and_reuses_reference_for_final_selection(self):
        candidate = self.candidate((200, 40, 50), "a")
        with patch.object(main, "initialize_database"), patch.object(main, "print_itinerary"), \
                patch.object(main.ObjectiveBalance, "from_data", wraps=objectives.ObjectiveBalance.from_data) as derive, \
                patch.object(main, "optimize", return_value=[candidate]) as optimize, \
                patch.object(main, "select_best_options", wraps=main.select_best_options) as select, \
                contextlib.redirect_stdout(io.StringIO()):
            main.main()
        derive.assert_called_once_with()
        self.assertIs(optimize.call_args.kwargs["balance"], select.call_args.args[1])

    def test_config_validation_needs_no_data_or_normalization(self):
        self.install([])
        with patch.object(objectives.ObjectiveBalance, "from_data", side_effect=AssertionError("Data requested")):
            main.validate_config()
        with self.assertRaisesRegex(ValueError, "requires a balance reference"):
            QualityMonitor().observe(ParetoArchive())

    def test_shared_clock_windows_preserve_daylight_saving_hours(self):
        calendar = {}
        with patch.multiple(config, SLEEP_START_HOUR=23, SLEEP_END_HOUR=7):
            for day, sleep, night in ((date(2027, 2, 1), 8, 6), (date(2027, 3, 27), 7, 5),
                                      (date(2027, 10, 30), 9, 7)):
                start = local_time(day, 22, "London")
                end = local_time(day + timedelta(days=1), 9, "London")
                with self.subTest(day=day):
                    self.assertEqual(sleep_overlap(start, end, "London", calendar), sleep)
                    self.assertEqual(sleep_overlap(start, end, "London"), sleep)
                    self.assertEqual(clock_window_overlap(start, end, "London", 0, 6, calendar), night)
                    self.assertEqual(clock_window_overlap(start, end, "London", 0, 6), night)
        for conn in self.connections:
            self.assertEqual(objectives.connection_objectives(conn, calendar), objectives.connection_objectives(conn))


if __name__ == "__main__":
    unittest.main()
