import contextlib
from dataclasses import replace
import io
import random
import unittest
from unittest.mock import patch

from test_optimizer import connection, genome
import config
import data_loader
import decoder
import main
import nsga2
import objectives
from accommodation import HotelFrontiers
from models import Chromosome, Hotel


class GeneticEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, OPTIMIZE_HOTELS=True,
            NUM_DESTINATIONS=1, TOTAL_DAYS=3, TRAVEL_RANGE=["2026-12-10", "2026-12-12"],
            MIN_DAYS_PER_CITY=2, MIN_NIGHTS_BY_CITY={}, MUST_VISIT_DESTINATIONS=[],
            MIN_USABLE_HOURS_PER_CITY=9, MAX_BUDGET=800, MAX_CO2=800, MAX_TRANSFER_HOURS=24,
            QUALITY_HV_MIN_GAIN=None, QUALITY_COMPROMISE_MIN_GAIN=None,
            COORDINATED_MUTATION_RATE=0.0,
        ))
        self.stack.enter_context(patch.multiple(nsga2, NUM_DESTINATIONS=1, TOTAL_AVAILABLE_DESTINATIONS=1))
        self.stack.enter_context(patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True))
        self.hotels = [
            Hotel(1, 1, {0: 20, 1: 20}, 8, 10, hotel_name="A budget", property_token="a"),
            Hotel(1, 3, {0: 40, 1: 40}, 1, 15, hotel_name="B comfort", property_token="b"),
        ]
        for module in (data_loader, main, nsga2, objectives):
            self.stack.enter_context(patch.object(module, "HOTELS_DB", self.hotels))
        self.outbound = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00")
        self.return_a = connection(1, 0, "2026-12-12T14:00+00:00", "2026-12-12T17:00+01:00",
                                   cost=20, co2=20, journey_id="a")
        self.return_b = replace(self.return_a, cost=45, co2=7, journey_id="b")
        self.install(self.outbound, self.return_a, self.return_b)

    def install(self, *connections):
        for conn in connections:
            data_loader.CONNECTIONS_BY_ROUTE.setdefault((conn.day, conn.src, conn.dest), []).append(conn)

    def individual(self, hotel=0.0, returning=0.0):
        ind = genome(choices=(0.0, returning))
        ind.hotel_choice = [hotel]
        return ind

    def test_one_input_is_one_evaluation_with_inherited_hotel_and_return_choices(self):
        batch = [self.individual(), self.individual(1.0, 1.0)]
        with patch.object(nsga2, "evaluate", wraps=objectives.evaluate) as evaluate:
            nsga2.evaluate_population(batch)
        self.assertEqual(len(batch), 2)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual([p.phenotype.hotels[0].property_token for p in batch], ["a", "b"])
        self.assertEqual([p.phenotype.connections[-1].journey_id for p in batch], ["a", "b"])
        for ind in batch:
            self.assertEqual(ind.constraint_violation, 0, ind.violations)
            self.assertEqual(objectives.check_solution(ind.phenotype), [])
            self.assertEqual(objectives.objective_values(ind.phenotype), ind.objectives)

    def test_initialization_is_random_single_pass(self):
        catalog = HotelFrontiers(self.hotels)
        search = decoder.TransportSearch()
        self.assertIsNone(search.stop_at)
        random.seed(42)
        population = nsga2.initialize_population(100, catalog, search)
        self.assertEqual(len(population), 100)
        self.assertEqual(search.generated, 100)
        self.assertGreater(len({p.hotel_choice[0] for p in population}), 90)

    def test_cached_decode_matches_standalone_greedy_and_is_repeatable(self):
        catalog = HotelFrontiers(self.hotels)
        search = decoder.TransportSearch()
        for returning in (0, 0.2, 0.8, 1):
            ind = self.individual(0.5, returning)
            expected = decoder.decode(ind)
            batch = [ind]
            nsga2.evaluate_population(batch, catalog, search)
            self.assertEqual(decoder.transport_key(ind.phenotype), decoder.transport_key(expected))
            again = [nsga2.clone(ind)]
            nsga2.evaluate_population(again, catalog, search)
            self.assertEqual(ind.objectives, again[0].objectives)
        self.assertGreater(search.cache_hits, 0)

    def test_hotel_cache_and_clones_do_not_share_mutable_selected_stays(self):
        catalog = HotelFrontiers(self.hotels)
        search = decoder.TransportSearch()
        first = self.individual()
        nsga2.evaluate_population([first], catalog, search)
        child = nsga2.clone(first)
        child.hotel_choice[0] = 1.0
        self.assertEqual(first.hotel_choice, [0.0])
        first.phenotype.hotels[0].total_cost = 999
        second = self.individual()
        nsga2.evaluate_population([second], catalog, search)
        self.assertEqual(second.phenotype.hotels[0].total_cost, 40)

    def test_crossover_preserves_city_linked_hotel_genes(self):
        first = Chromosome([1, 2, 3], [2, 2, 2], [9] * 4, [1] * 4, [0] * 4,
                           hotel_choice=[0.1, 0.2, 0.3])
        second = Chromosome([3, 1, 2], [3, 3, 3], [15] * 4, [2] * 4, [1] * 4,
                            hotel_choice=[0.6, 0.4, 0.5])
        random.seed(42)
        with patch.object(config, "TOTAL_DAYS", 12), patch.object(config, "CROSSOVER_RATE", 1):
            for _ in range(20):
                for child in nsga2.crossover(first, second):
                    for city, nights, hotel in zip(child.route, child.nights, child.hotel_choice):
                        choices = {(parent.nights[parent.route.index(city)],
                                    parent.hotel_choice[parent.route.index(city)])
                                   for parent in (first, second)}
                        self.assertIn((nights, hotel), choices)

    def test_mutation_evolves_hotels_and_handles_replaced_destinations(self):
        ind = self.individual()
        random.seed(42)
        with patch.object(config, "MUTATION_RATE", 1), patch.object(nsga2, "TOTAL_AVAILABLE_DESTINATIONS", 2):
            nsga2.mutate(ind)
        self.assertEqual(ind.route, [2])
        self.assertEqual(len(ind.hotel_choice), 1)
        self.assertTrue(0 <= ind.hotel_choice[0] <= 1)
        self.assertNotEqual(ind.hotel_choice, [0.0])

    def test_invalid_hotel_genes_are_rejected(self):
        for genes in ([], [0, 1], [-0.1], [1.1], [float("nan")]):
            with self.subTest(genes=genes):
                ind = self.individual()
                ind.hotel_choice = genes
                with self.assertRaisesRegex(ValueError, "Hotel-choice genes"):
                    nsga2.evaluate_population([ind])

    def test_no_hotels_ignores_a_supplied_catalog(self):
        catalog = HotelFrontiers(self.hotels)
        with patch.object(config, "OPTIMIZE_HOTELS", False):
            search = decoder.TransportSearch()
            ind = genome()
            nsga2.evaluate_population([ind], catalog, search)
        self.assertFalse(ind.phenotype.hotels)
        self.assertFalse(ind.hotel_choice)

    def test_hard_rules_still_reject_unavailable_hotels_and_resource_excess(self):
        for change in ("missing-night", "budget", "carbon", "daytime", "duration"):
            with self.subTest(change=change), contextlib.ExitStack() as stack:
                if change == "missing-night":
                    hotels = [replace(h, daily_rates={0: 20}) for h in self.hotels]
                    stack.enter_context(patch.object(nsga2, "HOTELS_DB", hotels))
                else:
                    name, value = {
                        "budget": ("MAX_BUDGET", 50), "carbon": ("MAX_CO2", 1),
                        "daytime": ("MIN_USABLE_HOURS_PER_CITY", 100),
                        "duration": ("MAX_TRANSFER_HOURS", 1),
                    }[change]
                    stack.enter_context(patch.object(config, name, value))
                ind = self.individual()
                nsga2.evaluate_population([ind])
                self.assertGreater(ind.constraint_violation, 0)
                self.assertTrue(ind.violations)

    def test_optimizer_runs_evolution_and_reports_separate_initial_and_offspring_counts(self):
        metrics = {}
        with patch.multiple(main, POPULATION_SIZE=8, MAX_GENERATIONS=5), \
                patch.object(config, "MUTATION_RATE", 1), \
                patch.object(main, "crossover", wraps=nsga2.crossover) as crossover, \
                patch.object(main, "mutate", wraps=nsga2.mutate) as mutate, \
                contextlib.redirect_stdout(io.StringIO()):
            front = main.optimize(metrics)
        self.assertEqual(metrics["generations"], 5)
        self.assertEqual(metrics["offspring_candidates"], 40)
        self.assertGreaterEqual(metrics["initialization_candidates"], 8)
        self.assertEqual(crossover.call_count, 20)
        self.assertEqual(mutate.call_count, 40)
        self.assertTrue(front)
        self.assertTrue(all(not objectives.check_solution(p.phenotype) for p in front))
        balance = objectives.ObjectiveBalance(tuple(metrics["balanced_solution_scales"]),
                                              tuple(metrics["balanced_solution_weights"]))
        selected = main.select_best_options(front, balance)["Balanced Optimum"]
        self.assertEqual(metrics["compromise"], balance.score(selected.objectives))

    def test_balanced_weights_do_not_change_genetic_exploration(self):
        fronts = []
        with patch.multiple(main, POPULATION_SIZE=8, MAX_GENERATIONS=3), \
                patch.multiple(config, QUALITY_HV_MIN_GAIN=None, QUALITY_COMPROMISE_MIN_GAIN=None), \
                contextlib.redirect_stdout(io.StringIO()):
            for weights in ((1, 0, 0), (0, 0, 1)):
                with patch.object(config, "BALANCED_SOLUTION_WEIGHTS", weights):
                    fronts.append(main.optimize())
        self.assertTrue(fronts[0])
        self.assertEqual([(p.objectives, p.phenotype) for p in fronts[0]],
                         [(p.objectives, p.phenotype) for p in fronts[1]])

if __name__ == "__main__":
    unittest.main()
