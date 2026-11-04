import contextlib
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import product
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from test_optimizer import connection, genome
import benchmark
import config
import data_loader
import decoder
import main
import nsga2
import objectives
from accommodation import HotelFrontiers
from models import Hotel, Phenotype
from timing import city_zone, make_visit, travel_window


class TransportSearchTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, NUM_DESTINATIONS=1, TOTAL_DAYS=4, TRAVEL_RANGE=["2026-12-10", "2026-12-16"],
            MIN_DAYS_PER_CITY=2, MIN_NIGHTS_BY_CITY={}, MUST_VISIT_DESTINATIONS=[], MIN_USABLE_HOURS_PER_CITY=0,
            OPTIMIZE_HOTELS=False, MAX_BUDGET=2000, MAX_CO2=2000,
            COORDINATED_MUTATION_RATE=0.2, PRUNE_EQUIVALENT_TRANSPORT=True, TRANSPORT_CACHE_SIZE=100))
        self.stack.enter_context(patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True))
        self.hotels = []
        for module in (nsga2, main, objectives):
            self.stack.enter_context(patch.object(module, "HOTELS_DB", self.hotels))
        self.stack.enter_context(patch.multiple(nsga2, NUM_DESTINATIONS=1, TOTAL_AVAILABLE_DESTINATIONS=1))

    def install(self, *connections):
        for conn in connections:
            data_loader.CONNECTIONS_BY_ROUTE.setdefault((conn.day, conn.src, conn.dest), []).append(conn)

    def dead_end(self):
        bad = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00", cost=10, co2=1)
        good = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-11T07:00+00:00")
        home = connection(1, 0, "2026-12-13T10:00+00:00", "2026-12-13T14:00+01:00")
        self.install(bad, good, home)
        return bad, good, home

    def run_candidate(self, ind=None, search=None):
        hotels = HotelFrontiers(self.hotels) if config.OPTIMIZE_HOTELS else None
        search = search or decoder.TransportSearch()
        batch = [ind or genome()]
        nsga2.evaluate_population(batch, hotels, search)
        return batch, search

    def test_dead_end_is_not_repaired_but_overnight_gene_remains_selectable(self):
        bad, good, home = self.dead_end()
        batch, search = self.run_candidate()
        self.assertEqual(batch[0].phenotype.connections, [bad])
        self.assertGreater(batch[0].constraint_violation, 0)
        batch, _ = self.run_candidate(genome(choices=(1.0, 0.0)), search)
        self.assertEqual(batch[0].phenotype.connections, [good, home])
        self.assertEqual(objectives.check_solution(batch[0].phenotype), [])
        self.assertEqual(search.pruned, 0)

    def test_wallclock_limit_reports_unknown_feasibility(self):
        self.dead_end()
        batch, search = self.run_candidate(search=decoder.TransportSearch(stop_at=0))
        self.assertEqual(batch[0].phenotype.search_status, "runtime_limit")
        self.assertIn("feasibility is unknown", " ".join(batch[0].violations))
        self.assertGreater(batch[0].constraint_violation, 0)
        self.assertFalse(search.result_cache)

    def test_encoded_genes_reproduce_connections_without_sharing_visits(self):
        self.dead_end()
        batch, search = self.run_candidate(genome(choices=(1.0, 0.0)))
        original = batch[0]
        expected = decoder.transport_key(original.phenotype)
        self.assertEqual(decoder.transport_key(decoder.decode(nsga2.clone(original))), expected)
        original.phenotype.visits[0].rest_nights = 99
        batch, _ = self.run_candidate(nsga2.clone(original), search)
        self.assertEqual(decoder.transport_key(batch[0].phenotype), expected)
        self.assertEqual(batch[0].phenotype.visits[0].rest_nights, 2)

    def test_safe_pruning_retains_objective_tradeoffs_time_changes_and_identity_ties(self):
        a = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00")
        dominated = replace(a, cost=30, co2=20, journey_id="dominated")
        tradeoff = replace(a, cost=30, co2=5, journey_id="tradeoff")
        equal = replace(a, journey_id="equal")
        later = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-11T07:00+00:00", journey_id="later")
        self.install(a, dominated, tradeoff, equal, later)
        search = decoder.TransportSearch()
        self.assertEqual({o.connection.journey_id for o in search.options(0, 0, 1)},
                         {"", "tradeoff", "equal", "later"})
        self.assertEqual(search.pruned, 1)
        with patch.object(config, "PRUNE_EQUIVALENT_TRANSPORT", False):
            self.assertEqual(len(decoder.TransportSearch().options(0, 0, 1)), 5)

    def test_cache_is_bounded_run_local_and_can_be_disabled(self):
        self.dead_end()
        search = decoder.TransportSearch()
        first = search.options(0, 0, 1)
        self.assertIs(search.options(0, 0, 1), first)
        self.assertEqual(search.cache_hits, 1)
        search.cache_size = 2
        for day in range(8):
            search.memo(("test", day), lambda: day)
        self.assertEqual(len(search.cache), 2)
        self.assertEqual(search.cache_evictions, 6)
        data_loader.CONNECTIONS_BY_ROUTE.clear()
        self.assertFalse(decoder.TransportSearch().options(0, 0, 1))
        with patch.object(config, "TRANSPORT_CACHE_SIZE", 0):
            uncached = decoder.TransportSearch()
            uncached.options(0, 0, 1)
            uncached.options(0, 0, 1)
            self.assertFalse(uncached.cache)
            self.assertFalse(uncached.option_table)
            self.assertEqual(uncached.cache_hits, 0)

    def test_option_order_uses_elapsed_time_across_dst_fold(self):
        with patch.object(config, "TRAVEL_RANGE", ["2026-10-25", "2026-10-30"]):
            early = connection(0, 1, "2026-10-25T02:30+02:00", "2026-10-25T05:00+00:00",
                               journey_id="early")
            late = connection(0, 1, "2026-10-25T02:15+01:00", "2026-10-25T06:00+00:00",
                              journey_id="late")
            for conn in (early, late):
                conn.departure_time = conn.departure_time.astimezone(city_zone("Stuttgart"))
            self.install(late, early)
            options = decoder.TransportSearch().options(0, 0, 1)
        self.assertEqual([o.connection.journey_id for o in options], ["early", "late"])

    def test_negative_resource_data_fail_explicitly(self):
        bad, _, _ = self.dead_end()
        bad.cost = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            decoder.TransportSearch()

    def test_flight_options_use_provider_emissions_without_multiplier(self):
        flight = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00",
                            mode="flight", co2=100)
        self.install(flight)
        self.assertEqual(decoder.TransportSearch().options(0, 0, 1)[0].values[2], 100)

    def test_initialization_balances_dates(self):
        with patch.multiple(config, TOTAL_DAYS=3, TRAVEL_RANGE=["2026-12-10", "2026-12-14"]):
            for day in range(3):
                self.install(connection(0, 1, f"2026-12-{10 + day}T09:00+01:00",
                                        f"2026-12-{10 + day}T12:00+00:00"),
                             connection(1, 0, f"2026-12-{12 + day}T10:00+00:00",
                                        f"2026-12-{12 + day}T14:00+01:00"))
            random.seed(13)
            search = decoder.TransportSearch()
            population = nsga2.initialize_population(18, None, search)
        self.assertEqual(len(population), 18)
        self.assertEqual([sum(p.start_day == day for p in population) for day in range(3)], [6, 6, 6])
        self.assertEqual(search.generated, 18)
        self.assertTrue(all(p.constraint_violation == 0 for p in population))

    def test_metrics_track_repeated_transport_with_different_hotels(self):
        self.dead_end()
        self.hotels.extend([
            Hotel(1, 2, {1: 20, 2: 20}, 3, 10, hotel_name="Cheap"),
            Hotel(1, 3, {1: 30, 2: 30}, 1, 5, hotel_name="Comfort")])
        with patch.object(config, "OPTIMIZE_HOTELS", True):
            catalog = HotelFrontiers(self.hotels)
            search = decoder.TransportSearch()
            for key in (0.0, 1.0):
                batch = [replace(genome(choices=(1.0, 0.0)), hotel_choice=[key])]
                nsga2.evaluate_population(batch, catalog, search)
                self.assertEqual(len(batch), 1)
        self.assertEqual((search.generated, search.feasible, search.complete), (2, 2, 2))
        self.assertEqual(len(search.seen_transports), 1)

    def test_coordinated_mutation_changes_neighboring_edges_and_preserves_dimensions(self):
        ind = genome()
        with patch.object(config, "MUTATION_RATE", 0), \
                patch.object(config, "COORDINATED_MUTATION_RATE", 1):
            random.seed(19)
            nsga2.mutate(ind)
        self.assertEqual(ind.mode_pref[0], ind.mode_pref[1])
        self.assertEqual(ind.dep_time_pref[0], ind.dep_time_pref[1])
        self.assertNotEqual(ind.connection_choice, [0, 0])
        self.assertEqual(ind.route, [1])
        self.assertEqual(ind.nights, [2])

    def test_invalid_search_parameters_rejected(self):
        for key, value in (("TRANSPORT_CACHE_SIZE", 1.5), ("TRANSPORT_RESULT_CACHE_SIZE", -1),
                           ("COORDINATED_MUTATION_RATE", -1), ("TRANSPORT_GROUP_SELECTION", 1)):
            with self.subTest(key=key), patch.object(config, key, value):
                with self.assertRaises(ValueError):
                    main.validate_config()

    def test_small_random_timetables_produce_only_independently_feasible_results(self):
        rng = random.Random(108)
        with patch.multiple(config, NUM_DESTINATIONS=2, TOTAL_DAYS=6, OPTIMIZE_HOTELS=True,
                            MAX_BUDGET=220, MAX_CO2=130):
            for case in range(8):
                data_loader.CONNECTIONS_BY_ROUTE.clear()
                self.hotels[:] = [
                    Hotel(city, 2, {d: rate for d in range(7)}, discomfort, carbon,
                          hotel_name=f"{city}-{rate}")
                    for city in (1, 2) for rate, discomfort, carbon in ((10, 8, 15), (25, 1, 2))]
                by_edge = []
                for src, dest in ((0, 1), (1, 2), (2, 0)):
                    options = []
                    for day in ([0] if src == 0 else range(2, 6)):
                        for hour in (9, 16, 23):
                            dep = datetime.fromisoformat("2026-12-10T00:00+01:00") + timedelta(days=day, hours=hour)
                            arr = dep + timedelta(hours=rng.choice((3, 8, 16)))
                            options.append(connection(src, dest, dep.isoformat(), arr.isoformat(),
                                                      mode=rng.choice(("bus", "flight")), cost=rng.randint(10, 55),
                                                      co2=rng.randint(1, 40)))
                    self.install(*options)
                    by_edge.append(options)
                catalog = HotelFrontiers(self.hotels)
                expected = set()
                for legs in product(*by_edge):
                    visits = [make_visit(city, data_loader.CITIES[city], travel_window(legs[i])[1],
                                         travel_window(legs[i + 1])[0])
                              for i, city in enumerate((1, 2))]
                    if any(v.nights != 2 or v.rest_nights < 2 for v in visits):
                        continue
                    p = Phenotype([0, 1, 2, 0], [2, 2], list(legs), [], visits)
                    for keys in product((0.0, 1.0), repeat=2):
                        candidate = catalog.select(p, list(keys))
                        if not objectives.check_solution(candidate):
                            expected.add(objectives.objective_values(candidate))
                batch = [replace(genome((1, 2), (2, 2), tuple(rng.random() for _ in range(3))),
                                 hotel_choice=[rng.random(), rng.random()]) for _ in range(20)]
                nsga2.evaluate_population(batch, catalog, decoder.TransportSearch())
                actual = {p.objectives for p in batch if p.constraint_violation == 0}
                with self.subTest(case=case):
                    self.assertEqual(len(batch), 20)
                    self.assertTrue(actual.issubset(expected))


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_restores_configuration_and_refuses_overwrite(self):
        calls = []
        original = config.PRUNE_EQUIVALENT_TRANSPORT, config.RANDOM_SEED, config.MAX_RUNTIME_SECONDS
        original_population = main.POPULATION_SIZE
        reference = objectives.ObjectiveBalance((200, 100, 100), (0.5, 0.25, 0.25))
        def optimize(metrics, *, balance):
            self.assertIs(balance, reference)
            calls.append((config.RANDOM_SEED, config.PRUNE_EQUIVALENT_TRANSPORT,
                          config.MAX_RUNTIME_SECONDS, main.POPULATION_SIZE))
            metrics.update(objective_minima=None, hypervolume=0, feasible_transport=0,
                           generated_transport=1, elapsed_seconds=config.MAX_RUNTIME_SECONDS)
            return []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark, "load_offline"), patch.object(main, "optimize", side_effect=optimize), \
                patch.object(main.ObjectiveBalance, "from_data", return_value=reference) as derive, \
                patch.object(data_loader, "CACHE_DIR", Path(directory)), \
                contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "comparison.json"
            benchmark.run_comparison(output, 10, [11, 42], ["genetic", "no_pruning"], 181.1, population=8)
            document = json.loads(output.read_text())
            self.assertEqual(len(document["runs"]), 4)
            self.assertEqual(document["population_size"], 8)
            self.assertEqual(document["balanced_solution_scales"], list(reference.scales))
            self.assertEqual(document["balanced_solution_weights"], list(reference.weights))
            derive.assert_called_once_with()
            self.assertEqual(set(calls), {(seed, pruning, 10, 8) for seed in (11, 42)
                                         for pruning in (True, False)})
            self.assertTrue(all(not run["reached_climate_target"] for run in document["runs"]))
            with self.assertRaises(FileExistsError):
                benchmark.run_comparison(output, 10, [11], ["genetic"])
        self.assertEqual((config.PRUNE_EQUIVALENT_TRANSPORT, config.RANDOM_SEED, config.MAX_RUNTIME_SECONDS), original)
        self.assertEqual(main.POPULATION_SIZE, original_population)

    def test_benchmark_preserves_configured_trip_and_model(self):
        observed = []
        def optimize(metrics, *, balance):
            observed.append((config.MAX_TRANSFER_HOURS, config.MIN_USABLE_HOURS_PER_CITY,
                             config.MAX_BUDGET, config.MAX_CO2, config.TOTAL_DAYS))
            metrics.update(objective_minima=None, hypervolume=0, feasible_transport=0,
                           generated_transport=1, elapsed_seconds=1)
            return []
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "load_offline"), \
                patch.object(main.ObjectiveBalance, "from_data", return_value=objectives.ObjectiveBalance(
                    (200, 100, 100), (0.5, 0.25, 0.25))), \
                patch.object(main, "optimize", side_effect=optimize), contextlib.redirect_stdout(io.StringIO()), \
                patch.multiple(config, MAX_TRANSFER_HOURS=19, MIN_USABLE_HOURS_PER_CITY=7,
                               MAX_BUDGET=777, MAX_CO2=333, TOTAL_DAYS=9):
            benchmark.run_comparison(Path(directory) / "comparison.json", 1, [11],
                                     ["genetic", "coordinated", "transport_diversity"])
        self.assertEqual(observed, [(19, 7, 777, 333, 9)] * 3)

    def test_benchmark_blocks_providers_on_cache_miss(self):
        for name in ("search_flights", "search_hotels", "EuropeTransitScraper"):
            with self.subTest(name=name), patch.object(data_loader, name) as provider, \
                    patch.object(data_loader, "initialize_database",
                                 side_effect=lambda: getattr(data_loader, name)()):
                with self.assertRaisesRegex(RuntimeError, "provider access is disabled"):
                    benchmark.load_offline()
                provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
