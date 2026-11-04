import sys
from pathlib import Path
for _sub in ('src', 'src/helpers', 'src/optimizer'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))
import contextlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import permutations, product
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
from models import Hotel, HotelStay, Phenotype, copy_phenotype
from progress import ParetoArchive
from timing import journey_eligible, local_time, make_visit


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, NUM_DESTINATIONS=1, TOTAL_DAYS=4, TRAVEL_RANGE=["2026-12-10", "2026-12-17"],
            MIN_DAYS_PER_CITY=2, MIN_NIGHTS_BY_CITY={}, MUST_VISIT_DESTINATIONS=[], MIN_USABLE_HOURS_PER_CITY=9,
            MAX_TRANSFER_HOURS=24, OPTIMIZE_HOTELS=False, MAX_BUDGET=2000, MAX_CO2=2000,
            TRANSPORT_RESULT_CACHE_SIZE=2, TRANSPORT_GROUP_SELECTION=True))
        self.stack.enter_context(patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True))
        self.stack.enter_context(patch.multiple(nsga2, NUM_DESTINATIONS=1, TOTAL_AVAILABLE_DESTINATIONS=1))

    def install(self, *connections):
        for conn in connections:
            data_loader.CONNECTIONS_BY_ROUTE.setdefault((conn.day, conn.src, conn.dest), []).append(conn)

    def trip(self):
        outbound = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-11T07:00+00:00")
        home = connection(1, 0, "2026-12-13T10:00+00:00", "2026-12-13T14:00+01:00")
        self.install(outbound, home)
        return outbound, home

    def test_duration_boundaries_all_modes_use_full_utc_connection(self):
        for mode, hours in product(decoder.MODES, (20, 24 - 1 / 3600, 24, 24 + 1 / 3600, 24 + 40 / 60, 32 + 40 / 60)):
            with self.subTest(mode=mode, hours=hours):
                dep = datetime.fromisoformat("2026-12-10T09:00+01:00")
                arr = (dep.astimezone(timezone.utc) + timedelta(hours=hours))
                conn = connection(0, 1, dep.isoformat(), arr.isoformat(), mode=mode,
                                  stops=2, access_hours=2, egress_hours=1)
                # The policy must not trust a stale or rounded display duration.
                conn.duration = 2
                self.assertEqual(journey_eligible(conn), hours <= 24)

    def test_offsets_dst_and_invalid_timestamps(self):
        for departure, arrival, expected in (
                ("2026-10-24T09:00+02:00", "2026-10-25T08:00+01:00", True),
                ("2026-10-24T09:00+02:00", "2026-10-25T09:00+01:00", False),
                ("2026-03-28T09:00+01:00", "2026-03-29T10:00+02:00", True)):
            self.assertEqual(journey_eligible(connection(0, 1, departure, arrival)), expected)
        outbound, _ = self.trip()
        for conn in (replace(outbound, departure_time=None),
                     replace(outbound, arrival_time=outbound.departure_time),
                     replace(outbound, departure_time=datetime(2026, 12, 10))):
            with self.assertRaises(ValueError):
                journey_eligible(conn)

    def test_policy_applies_to_decoder_and_independent_verification(self):
        outbound, home = self.trip()
        long = replace(outbound, arrival_time=outbound.arrival_time + timedelta(hours=10),
                       cost=1, co2=1)
        self.install(long)
        search = decoder.TransportSearch()
        self.assertEqual(search.duration_rejections, 1)
        self.assertEqual([o.connection for o in search.options(0, 0, 1)], [outbound])
        self.assertEqual(decoder.decode(genome()).connections[0], outbound)
        p = search.decode_and_encode(genome())
        self.assertEqual(objectives.check_solution(p), [])
        forged = replace(p, connections=[long, home])
        self.assertTrue(any("Journey duration exceeds" in e for e in objectives.check_solution(forged)))
        data_loader.CONNECTIONS_BY_ROUTE[0, 0, 1] = [long]
        rebuilt = decoder.TransportSearch()
        self.assertFalse(rebuilt.options(0, 0, 1))
        self.assertTrue(decoder.decode(genome()).errors)

    def test_slow_bus_is_not_removed_beside_fast_flight(self):
        bus = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-11T04:00+00:00")
        flight = replace(bus, mode="flight", arrival_time=bus.departure_time + timedelta(hours=2))
        self.install(bus, flight)
        self.assertEqual(len(decoder.TransportSearch().options(0, 0, 1)), 2)

    def test_daytime_nine_hour_boundary_and_forged_records(self):
        with patch.multiple(config, TOTAL_DAYS=2, MIN_DAYS_PER_CITY=1):
            for minutes, expected in ((-1, False), (0, True), (1, True)):
                data_loader.CONNECTIONS_BY_ROUTE.clear()
                outbound = connection(0, 1, "2026-12-10T16:00+01:00", "2026-12-10T17:00+00:00")
                departure = datetime.fromisoformat("2026-12-11T15:00+00:00") + timedelta(minutes=minutes)
                home = connection(1, 0, departure.isoformat(), (departure + timedelta(hours=2)).isoformat())
                self.install(outbound, home)
                visit = make_visit(1, "London", outbound.arrival_time, home.departure_time)
                self.assertAlmostEqual(visit.usable_hours, 9 + minutes / 60)
                forged = replace(visit, usable_hours=999)
                p = Phenotype([0, 1, 0], [1], [outbound, home], [], [forged])
                errors = objectives.check_solution(p)
                self.assertEqual(any("Insufficient usable" in e for e in errors), not expected)
                self.assertIn("Incorrect stay summary in London", errors)
                batch = [genome(nights=(1,))]
                nsga2.evaluate_population(batch)
                self.assertEqual(batch[0].constraint_violation == 0, expected)

    def test_allowances_reduce_daytime_but_not_scheduled_duration(self):
        with patch.multiple(config, TOTAL_DAYS=2, MIN_DAYS_PER_CITY=1):
            outbound = connection(0, 1, "2026-12-10T16:00+01:00", "2026-12-10T17:00+00:00",
                                  egress_hours=0.5)
            home = connection(1, 0, "2026-12-11T15:00+00:00", "2026-12-11T18:00+01:00")
            self.install(outbound, home)
            batch = [genome(nights=(1,))]
            nsga2.evaluate_population(batch)
            self.assertGreater(batch[0].constraint_violation, 0)
            with patch.object(config, "MIN_USABLE_HOURS_PER_CITY", 0):
                self.assertFalse(decoder.decode(genome(nights=(1,))).errors)

    def test_success_cache_exact_keys_eviction_and_independent_mutable_results(self):
        self.trip()
        search = decoder.TransportSearch()
        initial = genome()
        batch = [nsga2.clone(initial)]
        nsga2.evaluate_population(batch, search=search)
        expected = copy_phenotype(batch[0].phenotype)
        batch[0].phenotype.connections[0].cost = 999
        batch[0].phenotype.visits[0].usable_hours = 999
        batch = [nsga2.clone(initial)]
        nsga2.evaluate_population(batch, search=search)
        self.assertEqual(batch[0].phenotype, expected)
        self.assertEqual((search.invocations, search.result_hits), (1, 1))
        for choice in (0.00001, 0.00002):
            ind = genome(choices=(choice, 0))
            nsga2.evaluate_population([ind], search=search)
        self.assertEqual(search.invocations, 3)
        self.assertEqual(len(search.result_cache), 2)
        self.assertEqual(search.result_evictions, 1)
        search.stop_at = 0
        batch = [genome(choices=(0.00002, 0))]
        nsga2.evaluate_population(batch, search=search)
        self.assertEqual(batch[0].phenotype.search_status, "runtime_limit")
        self.assertGreater(batch[0].constraint_violation, 0)
        self.assertEqual(search.result_hits, 1)

    def test_no_cache_and_cache_match_and_interrupted_results_are_not_retained(self):
        self.trip()
        results = []
        for size in (0, 2):
            with patch.object(config, "TRANSPORT_RESULT_CACHE_SIZE", size):
                search = decoder.TransportSearch()
                for _ in range(2):
                    batch = [genome()]
                    nsga2.evaluate_population(batch, search=search)
                    results.append((batch[0].phenotype, batch[0].objectives, decoder.genotype_key(batch[0])))
                self.assertEqual(search.result_hits, int(size > 0))
                limited = decoder.TransportSearch(stop_at=0)
                nsga2.evaluate_population([genome()], search=limited)
                self.assertFalse(limited.result_cache)
        self.assertTrue(all(result == results[0] for result in results))

    def test_hotel_selection_preserves_equal_identities_and_independent_copies(self):
        hotels = [Hotel(1, 2, {d: 20 for d in range(7)}, 3, 10, hotel_name=name, property_token=name)
                  for name in ("A", "B")]
        catalog = HotelFrontiers(hotels)
        for day in range(3):
            arrival = local_time(datetime(2026, 12, 10).date() + timedelta(days=day), 12, "London")
            visit = make_visit(1, "London", arrival, arrival + timedelta(days=2))
            p = Phenotype([0, 1, 0], [2], [], [], [visit])
            selected = [catalog.select(p, [key]) for key in (0.0, 1.0)]
            self.assertEqual({h.hotels[0].property_token for h in selected}, {"A", "B"})
            selected[0].visits[0].usable_hours = 999
            self.assertNotEqual(selected[1].visits[0].usable_hours, 999)

    def test_config_validation_and_policy_output(self):
        for name, invalid in (("MAX_TRANSFER_HOURS", (0, -1, float("nan"), float("inf"))),
                              ("MIN_USABLE_HOURS_PER_CITY", (-1, float("nan"), float("inf")))):
            for value in invalid:
                with self.subTest(name=name, value=value), patch.object(config, name, value):
                    with self.assertRaises(ValueError):
                        main.validate_config()
        self.trip()
        batch = [genome()]
        nsga2.evaluate_population(batch)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main.print_itinerary("Policy", batch[0])
        self.assertIn("London:", output.getvalue())
        self.assertIn("hours", output.getvalue())
        self.assertNotIn("minimum 9", output.getvalue())

    def test_group_tickets_do_not_multiply_with_hotels_and_archive_retains_variants(self):
        self.trip()
        batch = [genome()]
        nsga2.evaluate_population(batch)
        a = batch[0]
        variants = []
        for i in range(8):
            p = copy_phenotype(a.phenotype)
            p.hotels = [HotelStay(str(i), 2, 1, 2, 40, 20, 6, 1, property_token=str(i))]
            variants.append(replace(a, phenotype=p, objectives=(50 + i, 100 - i, 20)))
        b = replace(a, phenotype=copy_phenotype(a.phenotype), objectives=(200, 200, 200))
        b.phenotype.connections[0].journey_id = "other"
        pop = variants + [b]
        groups = nsga2.parent_groups(pop)
        self.assertEqual(sorted(map(len, groups)), [1, 8])
        nsga2.rank_population(pop)
        with patch.object(nsga2.random, "sample", wraps=random.sample) as sample:
            nsga2.tournament_selection(pop, groups)
            self.assertIs(sample.call_args.args[0], groups)
        archive = ParetoArchive()
        archive.update(pop)
        self.assertEqual(len(archive.front), 8)
        survivors = nsga2.select_survivors(pop, 2, search=decoder.TransportSearch(stop_at=0))
        self.assertEqual(len({decoder.transport_key(i.phenotype) for i in survivors}), 2)
        self.assertEqual(len(archive.front), 8)
        self.assertIs(nsga2.tournament_selection([a]), a)
        incomplete = [replace(a, connection_choice=[choice, 0], constraint_violation=1)
                      for choice in (0.00001, 0.00002)]
        self.assertEqual(len(nsga2.parent_groups(incomplete)), 2)

    def test_tiny_full_feasible_set_matches_independent_enumeration(self):
        with patch.multiple(config, NUM_DESTINATIONS=2, TOTAL_DAYS=4, MIN_DAYS_PER_CITY=1,
                            OPTIMIZE_HOTELS=True, CHECK_IN_HOUR=9, CHECK_OUT_HOUR=18,
                            MIN_USABLE_HOURS_PER_CITY=5, DAYTIME_END_HOUR=18,
                            DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR={"train": 0.0, "bus": 0.0, "train+bus": 0.0, "flight": 0.0},
                            NIGHT_TRAVEL_DISCOMFORT_PER_HOUR={"train": 0.0, "bus": 0.0, "train+bus": 0.0, "flight": 0.0},
                            TRAVEL_RANGE=["2026-12-10", "2026-12-14"], MAX_BUDGET=125, MAX_CO2=75):
            hotels = [Hotel(city, 2, {d: rate for d in range(5)}, discomfort, carbon,
                            hotel_name=f"{city}-{rate}", property_token=f"{city}-{rate}")
                      for city in (1, 2) for rate, discomfort, carbon in ((10, 3, 5), (15, 1, 2))]
            for day, (src, dest) in product(range(5), permutations((0, 1, 2), 2)):
                dep = local_time(datetime(2026, 12, 10).date() + timedelta(days=day), 10, data_loader.CITIES[src])
                arr = local_time(dep.date(), 14, data_loader.CITIES[dest])
                for cost, carbon in ((10, 15), (20, 3)):
                    self.install(connection(src, dest, dep.isoformat(), arr.isoformat(), cost=cost, co2=carbon))
            with patch.object(objectives, "HOTELS_DB", hotels):
                catalog = HotelFrontiers(hotels)
                expected, actual = set(), set()
                for start, route, nights in product((0, 1), permutations((1, 2)), ((1, 2), (2, 1))):
                    days = (start, start + nights[0], start + sum(nights))
                    edges = list(zip((0, *route), (*route, 0)))
                    alternatives = [data_loader.CONNECTIONS_BY_ROUTE[day, *edge] for day, edge in zip(days, edges)]
                    for legs, hotel_pair in product(product(*alternatives), product(*[
                            [h for h in hotels if h.city == city] for city in route])):
                        cost = sum(c.cost for c in legs) + sum(h.daily_rates[0] * n for h, n in zip(hotel_pair, nights))
                        carbon = sum(c.co2 for c in legs) + sum(h.co2_per_night * n for h, n in zip(hotel_pair, nights))
                        if cost > 125 or carbon > 75:
                            continue
                        # All fixture journeys are daytime buses; each stay has n-1 full days + 5 daytime hours.
                        if any(9 * (n - 1) + 5 < 5 for n in nights):
                            continue
                        discomfort = sum((c.arrival_time.astimezone(timezone.utc)
                                          - c.departure_time.astimezone(timezone.utc)).total_seconds() / 3600 * 5
                                         for c in legs)
                        discomfort += sum(h.discomfort * n for h, n in zip(hotel_pair, nights))
                        expected.add((cost, discomfort, carbon))
                    for keys, hotel_keys in product(product((0.0, 1.0), repeat=3),
                                                    product((0.0, 1.0), repeat=2)):
                        batch = [replace(genome(route, nights, keys), start_day=start,
                                         hotel_choice=list(hotel_keys))]
                        nsga2.evaluate_population(batch, catalog, decoder.TransportSearch())
                        actual.update(ind.objectives for ind in batch if ind.constraint_violation == 0)
                self.assertTrue(expected)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
