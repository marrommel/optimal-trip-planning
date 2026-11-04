import sys
from pathlib import Path
for _sub in ('src', 'src/helpers', 'src/optimizer'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))
import contextlib
from dataclasses import replace
from datetime import datetime, timedelta
import io
from itertools import product
import math
import os
import random
import unittest
from unittest.mock import patch

from test_optimizer import connection, genome
import benchmark
import config
import data_loader
import main
import nsga2
import objectives
from accommodation import HotelFrontiers, stay_values
from models import Hotel, Phenotype
from progress import MutationSchedule, OffspringStats, ParetoArchive, QualityMonitor, hypervolume
from timing import departure_days, elapsed_hours, make_visit, trip_bounds


class FlexibleDateTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, TOTAL_DAYS=3, NUM_DESTINATIONS=1, TRAVEL_RANGE=["2026-12-10", "2026-12-14"],
            OPTIMIZE_HOTELS=False, MIN_DAYS_PER_CITY=2, MIN_NIGHTS_BY_CITY={}, MUST_VISIT_DESTINATIONS=[],
            MAX_BUDGET=2000, MAX_CO2=2000, MIN_USABLE_HOURS_PER_CITY=0,
            QUALITY_HV_MIN_GAIN=None, QUALITY_COMPROMISE_MIN_GAIN=None))
        self.stack.enter_context(patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True))
        self.stack.enter_context(patch.multiple(nsga2, NUM_DESTINATIONS=1, TOTAL_AVAILABLE_DESTINATIONS=1))
        self.hotels = []
        for module in (nsga2, main, objectives):
            self.stack.enter_context(patch.object(module, "HOTELS_DB", self.hotels))

    def install_trip(self, start_day, cost=20, egress=0):
        base = datetime(2026, 12, 10) + timedelta(days=start_day)
        outbound = connection(0, 1, f"{base.date()}T09:00+01:00", f"{base.date()}T12:00+00:00", cost=cost)
        end = base + timedelta(days=2)
        inbound = connection(1, 0, f"{end.date()}T10:00+00:00", f"{end.date()}T14:00+01:00",
                             cost=cost, egress_hours=egress)
        for conn in (outbound, inbound):
            data_loader.CONNECTIONS_BY_ROUTE.setdefault((conn.day, conn.src, conn.dest), []).append(conn)
        return outbound, inbound

    def evaluated(self, start=0):
        population = [replace(genome(), start_day=start)]
        nsga2.evaluate_population(population)
        return population

    def test_all_departures_including_last_possible_day(self):
        self.assertEqual(list(departure_days()), [0, 1, 2])
        self.install_trip(2)
        ind = self.evaluated(2)[0]
        self.assertEqual(ind.constraint_violation, 0, ind.violations)
        self.assertEqual([c.day for c in ind.phenotype.connections], [2, 4])
        start, end = trip_bounds(2)
        self.assertEqual(start.date().isoformat(), "2026-12-12")
        self.assertEqual(end.date().isoformat(), "2026-12-15")
        self.assertEqual(objectives.check_solution(ind.phenotype), [])

    def test_departure_outside_window_is_rejected(self):
        self.install_trip(2)
        for day in (-1, 3, 1.5, True):
            with self.subTest(day=day):
                ind = self.evaluated(day)[0]
                self.assertGreater(ind.constraint_violation, 0)
                self.assertIn("Departure date", " ".join(ind.violations))

    def test_early_return_rejected_by_decoder_and_independent_verifier(self):
        self.install_trip(0)
        valid = self.evaluated()[0].phenotype
        with patch.object(config, "TOTAL_DAYS", 4):
            self.assertGreater(self.evaluated()[0].constraint_violation, 0)
            self.assertIn("Home return must be on the exact last calendar day", objectives.check_solution(valid))

    def test_egress_controls_actual_return_day(self):
        self.install_trip(0, egress=12)
        self.assertGreater(self.evaluated()[0].constraint_violation, 0)
        with patch.object(config, "TOTAL_DAYS", 4):
            ind = self.evaluated()[0]
            self.assertEqual(ind.constraint_violation, 0, ind.violations)
        self.install_trip(2, egress=12)
        self.assertGreater(self.evaluated(2)[0].constraint_violation, 0)

    def test_verifier_rejects_tampered_start_date(self):
        self.install_trip(2)
        p = self.evaluated(2)[0].phenotype
        errors = objectives.check_solution(replace(p, start_day=0))
        self.assertIn("First departure must be on the selected start date", errors)

    def test_exact_duration_does_not_fit_window(self):
        with patch.object(config, "TOTAL_DAYS", 6), patch.object(main, "initialize_database") as load:
            with self.assertRaisesRegex(ValueError, "too short"):
                main.main()
            load.assert_not_called()

    def test_calendar_duration_across_dst_is_not_fixed_elapsed_hours(self):
        with patch.object(config, "TRAVEL_RANGE", ["2026-10-24", "2026-10-28"]):
            start, end = trip_bounds()
            self.assertEqual((end.date() - start.date()).days, 3)
            self.assertEqual(elapsed_hours(start, end), 73)

    def test_later_start_uses_absolute_hotel_quote_dates(self):
        self.install_trip(2)
        self.hotels.append(Hotel(1, 2, {2: 31, 3: 47}, 3, 15, hotel_name="Later stay"))
        with patch.object(config, "OPTIMIZE_HOTELS", True):
            ind = self.evaluated(2)[0]
            self.assertEqual(ind.constraint_violation, 0, ind.violations)
            stay = ind.phenotype.hotels[0]
            self.assertEqual((stay.check_in_day, stay.nights, stay.total_cost), (2, 2, 78))

    def test_date_genes_survive_cloning_crossover_and_mutation(self):
        random.seed(7)
        population = nsga2.init_population(9)
        self.assertEqual({p.start_day for p in population}, {0, 1, 2})
        a, b = replace(genome(), start_day=0), replace(genome(), start_day=2)
        self.assertEqual(nsga2.clone(b).start_day, 2)
        with patch.object(config, "CROSSOVER_RATE", 1):
            for _ in range(20):
                self.assertTrue(all(c.start_day in (0, 2) for c in nsga2.crossover(a, b)))
        with patch.object(config, "MUTATION_RATE", 1):
            seen = set()
            for _ in range(30):
                nsga2.mutate(a)
                seen.add(a.start_day)
            self.assertEqual(seen, {0, 1, 2})

    def test_search_discovers_cheaper_later_departure(self):
        self.install_trip(0, cost=100)
        self.install_trip(2, cost=20)
        with patch.multiple(main, POPULATION_SIZE=6, MAX_GENERATIONS=2), contextlib.redirect_stdout(io.StringIO()):
            front = main.optimize()
        self.assertTrue(front)
        self.assertEqual(min(p.objectives[0] for p in front), 40)
        self.assertTrue(all(p.start_day == 2 for p in front))
        self.assertTrue(all(not objectives.check_solution(p.phenotype) for p in front))

    def test_archive_retains_hotel_front_larger_than_population(self):
        self.install_trip(0)
        self.hotels.extend(Hotel(1, 2, {0: 20 + i * 10, 1: 20 + i * 10},
                                 10 - i, 15, hotel_name=f"Hotel {i}", property_token=str(i))
                           for i in range(5))
        def initialize(size, hotels, search, record):
            batch = [replace(genome(), hotel_choice=[i / 4]) for i in range(5)]
            nsga2.evaluate_population(batch, hotels, search, record)
            return batch
        with patch.multiple(config, OPTIMIZE_HOTELS=True, TRAVEL_RANGE=["2026-12-10", "2026-12-12"]), \
                patch.multiple(main, POPULATION_SIZE=2, MAX_GENERATIONS=1), \
                patch.object(main, "initialize_population", side_effect=initialize), \
                contextlib.redirect_stdout(io.StringIO()):
            front = main.optimize()
        self.assertEqual(len(front), 5)
        self.assertTrue(all(not objectives.check_solution(p.phenotype) for p in front))

    def test_constraints_applied_to_each_hotel_alternative(self):
        self.install_trip(0)
        self.hotels.extend([
            Hotel(1, 2, {0: 20, 1: 20}, 3, 100, hotel_name="Cheap high carbon"),
            Hotel(1, 3, {0: 40, 1: 40}, 1, 1, hotel_name="Comfortable low carbon")])
        with patch.multiple(config, OPTIMIZE_HOTELS=True, MAX_CO2=30):
            population = [replace(genome(), hotel_choice=[key]) for key in (0.0, 1.0)]
            nsga2.evaluate_population(population)
        self.assertEqual(len(population), 2)
        self.assertEqual(sum(p.constraint_violation == 0 for p in population), 1)
        self.assertEqual(next(p for p in population if not p.constraint_violation).phenotype.hotels[0].hotel_name,
                         "Comfortable low carbon")


class AccommodationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, TOTAL_DAYS=8, TRAVEL_RANGE=["2026-12-10", "2026-12-23"],
            MIN_DAYS_PER_CITY=2, MIN_NIGHTS_BY_CITY={}, ALLOW_SYNTHETIC_HOTELS=False))

    def hotel(self, name, city=1, cost=20, discomfort=3, co2=10, **kwargs):
        return Hotel(city, 2, {day: cost for day in range(13)}, discomfort, co2,
                     hotel_name=name, property_token=name, **kwargs)

    def transport(self, cities=(1,)):
        base = datetime.fromisoformat("2026-12-10T12:00+01:00")
        visits = [make_visit(city, data_loader.CITIES[city], base + timedelta(days=i * 2),
                             base + timedelta(days=(i + 1) * 2)) for i, city in enumerate(cities)]
        return Phenotype([0, *cities, 0], [2] * len(cities), [], [], visits)

    def test_local_dominance_preserves_comfort_climate_tradeoffs_and_identity_ties(self):
        hotels = [
            self.hotel("Cheap"), self.hotel("Dominated", cost=30),
            self.hotel("Comfort", cost=30, discomfort=1),
            self.hotel("Climate", cost=30, co2=1),
            self.hotel("Cheap equivalent")]
        catalog = HotelFrontiers(hotels)
        names = {h.hotel_name for h in catalog.options[1, 0, 2]}
        self.assertEqual(names, {"Cheap", "Comfort", "Climate", "Cheap equivalent"})
        selected = [catalog.select(self.transport(), [i / 3]) for i in range(4)]
        self.assertEqual({p.hotels[0].hotel_name for p in selected}, names)

    def test_hotel_genes_can_select_every_three_city_combination(self):
        hotels = [self.hotel(f"{city}-{i}", city, cost=10 + i * 5, discomfort=10 - i * 3, co2=5 + i)
                  for city in (1, 2, 3) for i in range(3)]
        catalog = HotelFrontiers(hotels)
        options = [catalog.options[city, i * 2, 2] for i, city in enumerate((1, 2, 3))]
        exhaustive = []
        for stays in product(*options):
            values = tuple(sum(stay_values(h)[axis] for h in stays) for axis in range(3))
            exhaustive.append((values, tuple(h.property_token for h in stays)))
        expected = set(exhaustive)
        selected = [catalog.select(self.transport((1, 2, 3)), list(keys))
                    for keys in product((0.0, 0.5, 1.0), repeat=3)]
        actual = {(tuple(sum(stay_values(h)[axis] for h in p.hotels) for axis in range(3)),
                   tuple(h.property_token for h in p.hotels)) for p in selected}
        self.assertEqual(actual, expected)

    def test_missing_nights_and_synthetic_hotels_do_not_produce_fallback_offers(self):
        incomplete = self.hotel("Missing night")
        del incomplete.daily_rates[1]
        catalog = HotelFrontiers([incomplete, self.hotel("Synthetic", synthetic=True)])
        p = catalog.select(self.transport(), [0.0])
        self.assertFalse(p.hotels)
        self.assertIn("No available hotel", p.errors[0])

    def test_snapshot_and_repeated_selection_do_not_leak_mutation_or_rng(self):
        hotel = self.hotel("Snapshot")
        before = random.getstate()
        catalog = HotelFrontiers([hotel])
        p = self.transport()
        first = catalog.select(p, [0.0])
        first.hotels[0].total_cost = 999
        hotel.daily_rates[0] = 100
        self.assertEqual(catalog.select(p, [0.0]).hotels[0].total_cost, 40)
        self.assertEqual(HotelFrontiers([hotel]).select(p, [0.0]).hotels[0].total_cost, 120)
        self.assertEqual(before, random.getstate())

    def test_invalid_hotel_contributions_fail_explicitly(self):
        for field in ("cost", "discomfort", "co2"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                        HotelFrontiers([self.hotel("Invalid", **{field: value})])


class QualityTests(unittest.TestCase):
    def candidate(self, values=(50, 50, 50), identifier="a", violation=0):
        ind = genome()
        outbound = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00",
                              journey_id=identifier)
        inbound = connection(1, 0, "2026-12-12T10:00+00:00", "2026-12-12T14:00+01:00")
        ind.phenotype = Phenotype([0, 1, 0], [2], [outbound, inbound], [])
        ind.objectives = values
        ind.constraint_violation = violation
        return ind

    def monitor(self, **kwargs):
        with patch.multiple(config, QUALITY_SCALES=(100, 100, 100), HYPERVOLUME_REFERENCE=(200, 200, 200),
                            QUALITY_WINDOW_GENERATIONS=2, QUALITY_HV_MIN_GAIN=kwargs.get("hv", 0),
                            QUALITY_COMPROMISE_MIN_GAIN=kwargs.get("score", 0),
                            FRONT_PROGRESS_RESOLUTION=0.1):
            return QualityMonitor(objectives.ObjectiveBalance((200, 100, 100), (0.5, 0.25, 0.25)))

    def test_exact_hypervolume_with_overlaps_dominance_and_empty_front(self):
        self.assertEqual(hypervolume(set(), (5, 5, 5)), 0)
        self.assertEqual(hypervolume({(1, 3, 3)}, (5, 5, 5)), 16)
        self.assertEqual(hypervolume({(1, 3, 3), (3, 1, 3), (4, 4, 4)}, (5, 5, 5)), 24)

    def test_hypervolume_matches_exhaustive_box_union(self):
        rng = random.Random(13)
        for _ in range(20):
            points = {tuple(rng.randint(1, 8) for _ in range(3)) for _ in range(8)}
            axes = [sorted({p[i] for p in points} | {10}) for i in range(3)]
            expected = 0
            for spans in product(*(list(zip(axis, axis[1:])) for axis in axes)):
                if any(all(p[i] <= spans[i][0] for i in range(3)) for p in points):
                    expected += math.prod(b - a for a, b in spans)
            self.assertEqual(hypervolume(points, (10, 10, 10)), expected)

    def test_invalid_or_outside_reference_points_fail_explicitly(self):
        for point in ((5, 1, 1), (6, 1, 1), (-1, 1, 1), (float("nan"), 1, 1)):
            with self.assertRaises(ValueError):
                hypervolume({point}, (5, 5, 5))

    def test_archive_keeps_equal_objectives_distinct_itineraries_and_rejects_infeasible(self):
        archive = ParetoArchive()
        a, b = self.candidate(), self.candidate(identifier="b")
        archive.update([a, a, b, self.candidate((1, 1, 1), "invalid", violation=1)])
        self.assertEqual(len(archive.front), 2)
        archive.update([self.candidate((40, 40, 40), "better")])
        self.assertEqual(len(archive.front), 1)
        self.assertEqual(len(archive.keys), 1)

    def test_fixed_normalization_does_not_follow_front_or_config_changes(self):
        monitor, archive = self.monitor(), ParetoArchive()
        archive.update([self.candidate()])
        initial = monitor.observe(archive)
        self.assertAlmostEqual(initial.hypervolume, 1.5 ** 3)
        with patch.multiple(config, QUALITY_SCALES=(1, 1, 1), HYPERVOLUME_REFERENCE=(2, 2, 2),
                            BALANCED_SOLUTION_WEIGHTS=(0, 0, 1)):
            repeated = monitor.observe(archive)
        self.assertEqual(initial.hypervolume, repeated.hypervolume)
        self.assertEqual(initial.compromise, repeated.compromise)
        archive.update([self.candidate((20, 100, 100), "tradeoff")])
        self.assertGreater(monitor.observe(archive).hypervolume, initial.hypervolume)

    def test_stopping_requires_full_window_and_both_quality_metrics(self):
        archive = ParetoArchive()
        archive.update([self.candidate()])
        monitor = self.monitor(hv=100, score=0)
        self.assertFalse(monitor.observe(archive).stop)
        self.assertFalse(monitor.observe(archive).stop)
        archive.update([self.candidate((40, 40, 40), "improved")])
        self.assertFalse(monitor.observe(archive).stop)
        self.assertFalse(monitor.observe(archive).stop)
        self.assertTrue(monitor.observe(archive).stop)

    def test_stopping_is_disabled_until_calibrated_and_empty_archive_never_stops(self):
        archive = ParetoArchive()
        monitor = self.monitor()
        for _ in range(5):
            self.assertFalse(monitor.observe(archive).stop)
        disabled = self.monitor(hv=None, score=None)
        archive.update([self.candidate()])
        for _ in range(5):
            self.assertFalse(disabled.observe(archive).stop)

    def test_regions_distinguish_equal_cells_from_material_improvement(self):
        archive, monitor = ParetoArchive(), self.monitor()
        archive.update([self.candidate((55, 55, 55))])
        self.assertEqual(monitor.observe(archive).improved_regions, 1)
        monitor.reset_report_counts()
        archive.update([self.candidate((54, 54, 54), "same cell")])
        within = monitor.observe(archive)
        self.assertEqual(within.archive_additions, 1)
        self.assertEqual(within.improved_regions, 0)
        archive.update([self.candidate((44, 54, 54), "better cell")])
        self.assertEqual(monitor.observe(archive).improved_regions, 1)

    def test_offspring_rates_count_decoded_alternatives_not_population_survivors(self):
        a, b = self.candidate(), self.candidate(identifier="b", violation=1)
        with patch.object(config, "OPTIMIZE_HOTELS", False):
            stats = OffspringStats.measure([a, a, b], generated=2)
        self.assertEqual((stats.generated, stats.decoded, stats.unique_feasible), (2, 3, 1))
        self.assertEqual((stats.duplicate_complete, stats.complete), (1, 3))

    def test_progress_omits_diagnostics_on_narrow_terminals(self):
        a = self.candidate()
        archive, monitor = ParetoArchive(), self.monitor()
        archive.update([a])
        quality = monitor.observe(archive)
        before = random.getstate()
        output = io.StringIO()
        with patch.object(config, "OPTIMIZE_HOTELS", False), \
                patch.object(main, "get_terminal_size", return_value=os.terminal_size((40, 24))), \
                contextlib.redirect_stdout(output):
            main.print_optimization_progress([a, a], archive.front, 0, 0)
        text = output.getvalue()
        compact = "".join(text.split())
        for label in ("Gen", "Best EUR", "CO2 kg", "Discomfort"):
            self.assertIn("".join(label.split()), compact)
        for label in ("hypervolume", "offspring", "diversity", "regions"):
            self.assertNotIn(label, text)
        self.assertEqual(len(text.splitlines()), 4)
        self.assertLessEqual(max(map(len, text.splitlines())), 40)
        self.assertEqual(before, random.getstate())

    def test_invalid_quality_configuration_is_rejected(self):
        for changes in ({"QUALITY_WINDOW_GENERATIONS": 0}, {"QUALITY_SCALES": (0, 1, 1)},
                        {"HYPERVOLUME_REFERENCE": (1, 2)}, {"QUALITY_HV_MIN_GAIN": -1},
                        {"QUALITY_HV_MIN_GAIN": 0, "QUALITY_COMPROMISE_MIN_GAIN": None},
                        {"BALANCED_SOLUTION_WEIGHTS": (0, 0, 0)},
                        {"BALANCED_SOLUTION_WEIGHTS": (-1, 1, 1)},
                        {"BALANCED_SOLUTION_WEIGHTS": (1, 1)}):
            with self.subTest(changes=changes), patch.multiple(config, **changes):
                with self.assertRaises(ValueError):
                    QualityMonitor()


class AdaptiveMutationTests(unittest.TestCase):
    def schedule(self, **changes):
        defaults = dict(MUTATION_RATE=0.1, MUTATION_RATE_MIN=0.02, MUTATION_RATE_MAX=0.4,
                        STEP_GROWTH=1.25, SUCCESS_TARGET=0.2)
        self.stack.enter_context(patch.multiple(config, **(defaults | changes)))
        return MutationSchedule()

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def test_step_grows_while_offspring_improve_and_shrinks_once_they_stop(self):
        schedule = self.schedule()
        self.assertEqual((schedule.rate, schedule.coarse), (0.1, False))
        schedule.observe(3, 10)
        self.assertAlmostEqual(schedule.rate, 0.125)
        self.assertTrue(schedule.coarse)
        self.assertAlmostEqual(schedule.success_ratio, 0.3)
        for _ in range(2):
            schedule.observe(0, 10)
        self.assertAlmostEqual(schedule.rate, 0.08)
        self.assertFalse(schedule.coarse)

    def test_rate_stays_inside_the_configured_bounds(self):
        schedule = self.schedule()
        for _ in range(50):
            schedule.observe(10, 10)
        self.assertAlmostEqual(schedule.rate, 0.4)
        for _ in range(100):
            schedule.observe(0, 10)
        self.assertAlmostEqual(schedule.rate, 0.02)
        # A clamped bound must not wind up: one successful generation moves the rate again.
        schedule.observe(10, 10)
        self.assertAlmostEqual(schedule.rate, 0.025)

    def test_pinned_bounds_reproduce_a_fixed_step(self):
        schedule = self.schedule(MUTATION_RATE_MIN=0.1, MUTATION_RATE_MAX=0.1)
        for successes in (10, 0, 10, 0):
            schedule.observe(successes, 10)
            self.assertEqual((schedule.rate, schedule.coarse), (0.1, False))

    def test_empty_offspring_leaves_the_schedule_untouched(self):
        schedule = self.schedule()
        schedule.observe(0, 0)
        self.assertEqual((schedule.rate, schedule.success_ratio), (0.1, None))

    def test_coarse_steps_change_more_than_fine_steps(self):
        with patch.multiple(config, MUST_VISIT_DESTINATIONS=[], OPTIMIZE_HOTELS=False,
                            COORDINATED_MUTATION_RATE=0, MUTATION_RATE=1):
            spans = {}
            for coarse in (False, True):
                schedule = MutationSchedule()
                schedule.scale = 1.5 if coarse else 1.0
                schedule.rate = 1
                steps = set()
                random.seed(5)
                for _ in range(200):
                    ind = genome((1, 2), (3, 3), (0.1, 0.2, 0.3))
                    before = ind.nights[:]
                    nsga2.mutate(ind, schedule)
                    steps.update(abs(a - b) for a, b in zip(before, ind.nights))
                spans[coarse] = steps
        self.assertTrue(max(spans[True]) > max(spans[False]))

    def test_baseline_step_leaves_route_operators_unrestricted(self):
        """At scale 1.0 the operators must behave exactly as without a schedule;
        only a step below the baseline narrows insertion and inversion."""
        def large_span_count(scale):
            schedule = MutationSchedule()
            schedule.scale = scale
            schedule.rate = 1
            large = 0
            random.seed(11)
            for _ in range(300):
                ind = genome((1, 2, 3, 4, 5, 6), (1,) * 6, (0.1,) * 7)
                before = ind.route[:]
                nsga2.mutate(ind, schedule)
                moved = [i for i, (a, b) in enumerate(zip(before, ind.route)) if a != b]
                large += bool(moved) and moved[-1] - moved[0] > 2
            return large

        with patch.multiple(config, MUST_VISIT_DESTINATIONS=[], OPTIMIZE_HOTELS=False,
                            COORDINATED_MUTATION_RATE=0, MUTATION_RATE=1), \
                patch.multiple(nsga2, TOTAL_AVAILABLE_DESTINATIONS=6,
                               CITIES={i: f"City{i}" for i in range(7)}):
            baseline, fine = large_span_count(1.0), large_span_count(0.8)
        # Swap stays unrestricted, so fine mutation still moves distant cities occasionally.
        self.assertGreater(baseline, 2 * fine)
        self.assertGreater(fine, 0)

    def test_fine_and_coarse_are_exclusive_around_the_baseline(self):
        schedule = self.schedule()
        self.assertEqual((schedule.coarse, schedule.fine), (False, False))
        schedule.observe(3, 10)
        self.assertEqual((schedule.coarse, schedule.fine), (True, False))
        for _ in range(2):
            schedule.observe(0, 10)
        self.assertEqual((schedule.coarse, schedule.fine), (False, True))

    def test_benchmark_variants_can_pin_the_step(self):
        settings = benchmark.variants()["fixed_step"]
        self.assertEqual(settings["MUTATION_RATE_MIN"], config.MUTATION_RATE)
        self.assertEqual(settings["MUTATION_RATE_MAX"], config.MUTATION_RATE)

    def test_invalid_adaptive_configuration_is_rejected(self):
        for changes in ({"MUTATION_RATE_MIN": 0}, {"MUTATION_RATE_MAX": 1.5},
                        {"MUTATION_RATE_MIN": 0.5, "MUTATION_RATE_MAX": 0.4},
                        {"SUCCESS_TARGET": 0}, {"SUCCESS_TARGET": 1},
                        {"STEP_GROWTH": 1}, {"STEP_GROWTH": float("inf")}):
            with self.subTest(changes=changes), patch.multiple(config, **changes):
                with self.assertRaises(ValueError):
                    main.validate_config()


if __name__ == "__main__":
    unittest.main()
