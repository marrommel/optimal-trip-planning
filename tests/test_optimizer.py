import contextlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

for _sub in ("src", "src/helpers", "src/optimizer"): sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))

import config
import data_loader
import decoder
import main
import nsga2
import objectives
from models import Chromosome, Connection, Hotel, HotelStay, Phenotype
from timing import clock_window_overlap, elapsed_hours, make_visit, sleep_overlap, travel_window, trip_bounds


def connection(src, dest, departure, arrival, mode="bus", cost=20, co2=10, **kwargs):
    dep, arr = datetime.fromisoformat(departure), datetime.fromisoformat(arrival)
    day = (dep.date() - datetime.fromisoformat(config.TRAVEL_RANGE[0]).date()).days
    return Connection(
        day, src, dest, mode, dep.hour + dep.minute / 60, elapsed_hours(dep, arr), cost, co2,
        src_name=data_loader.CITIES[src], dest_name=data_loader.CITIES[dest],
        provider="fixture", departure_time=dep, arrival_time=arr, **kwargs)


def genome(route=(1,), nights=(2,), choices=(0.0, 0.0)):
    return Chromosome(list(route), list(nights), [9.0] * (len(route) + 1),
                      [1] * (len(route) + 1), list(choices),
                      hotel_choice=[0.0] * len(route) if config.OPTIMIZE_HOTELS else [])


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            config, NUM_DESTINATIONS=1, TOTAL_DAYS=3, TRAVEL_RANGE=["2026-12-10", "2026-12-23"],
            OPTIMIZE_HOTELS=False, MIN_DAYS_PER_CITY=2, MIN_NIGHTS_BY_CITY={}, MUST_VISIT_DESTINATIONS=[],
            MIN_USABLE_HOURS_PER_CITY=0, MAX_BUDGET=300, MAX_CO2=2000))
        self.stack.enter_context(patch.dict(data_loader.CONNECTIONS_BY_ROUTE, clear=True))
        self.stack.enter_context(patch.object(nsga2, "HOTELS_DB", []))
        self.stack.enter_context(patch.object(main, "HOTELS_DB", nsga2.HOTELS_DB))
        self.stack.enter_context(patch.object(objectives, "HOTELS_DB", nsga2.HOTELS_DB))
        self.outbound = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T17:00+00:00")
        self.inbound = connection(1, 0, "2026-12-12T10:00+00:00", "2026-12-12T20:00+01:00")

    def install(self, *connections):
        for conn in connections:
            data_loader.CONNECTIONS_BY_ROUTE.setdefault((conn.day, conn.src, conn.dest), []).append(conn)

    def evaluate(self, ind=None):
        ind = ind or genome()
        nsga2.evaluate_population([ind])
        return ind

    def test_six_calendar_days_end_before_december_16(self):
        self.stack.enter_context(patch.object(config, "TOTAL_DAYS", 6))
        start, deadline = trip_bounds()
        self.assertEqual(start.isoformat(), "2026-12-10T00:00:00+01:00")
        self.assertEqual(deadline.isoformat(), "2026-12-16T00:00:00+01:00")
        late = connection(1, 0, "2026-12-15T23:00+00:00", "2026-12-16T10:00+01:00")
        self.install(self.outbound, late)
        ind = self.evaluate(genome(nights=(5,)))
        self.assertGreater(ind.constraint_violation, 0)
        self.assertTrue(any("No time-feasible" in v for v in ind.violations))
        self.assertNotIn("Penalty", str(ind.phenotype))

    def test_egress_can_push_arrival_past_deadline(self):
        self.stack.enter_context(patch.object(config, "TOTAL_DAYS", 6))
        late = connection(1, 0, "2026-12-15T12:00+00:00", "2026-12-15T23:45+01:00", egress_hours=0.5)
        self.install(self.outbound, late)
        self.assertGreater(self.evaluate(genome(nights=(5,))).constraint_violation, 0)

    def test_overnight_arrival_moves_stay_and_next_departure(self):
        self.stack.enter_context(patch.object(config, "TOTAL_DAYS", 4))
        overnight = connection(0, 1, "2026-12-10T09:50+01:00", "2026-12-11T05:25+00:00")
        onward = replace(self.inbound, day=3,
                          departure_time=self.inbound.departure_time + timedelta(days=1),
                          arrival_time=self.inbound.arrival_time + timedelta(days=1))
        self.install(overnight, onward)
        ind = self.evaluate()
        self.assertEqual(ind.constraint_violation, 0, ind.violations)
        self.assertEqual(ind.phenotype.connections[1].day, 3)
        visit = ind.phenotype.visits[0]
        self.assertEqual(visit.nights, 2)
        self.assertEqual(visit.arrival.date().isoformat(), "2026-12-11")
        self.assertEqual(visit.rest_nights, 2)

    def test_early_return_does_not_count_an_unrested_night(self):
        early = connection(1, 0, "2026-12-12T01:00+00:00", "2026-12-12T10:00+01:00")
        self.install(self.outbound, early)
        self.assertGreater(self.evaluate().constraint_violation, 0)

    def test_city_minimum_and_required_destination(self):
        self.install(self.outbound, self.inbound)
        with patch.object(config, "MIN_NIGHTS_BY_CITY", {"London": 3}):
            self.assertGreater(self.evaluate().constraint_violation, 0)
        with patch.object(config, "MUST_VISIT_DESTINATIONS", ["Paris"]):
            self.assertIn("A required destination was omitted", self.evaluate().violations)

    def test_minimum_usable_daytime(self):
        self.install(self.outbound, self.inbound)
        with patch.object(config, "MIN_USABLE_HOURS_PER_CITY", 25):
            self.assertGreater(self.evaluate().constraint_violation, 0)

    def test_every_same_time_option_and_mixed_mode_are_selectable(self):
        dearer = replace(self.outbound, cost=30, co2=5, journey_id="second")
        mixed = replace(self.outbound, mode="train+bus", journey_id="mixed")
        self.install(self.outbound, dearer, mixed, self.inbound)
        cheap = self.evaluate(genome(choices=(0, 0)))
        expensive = self.evaluate(genome(choices=(1, 0)))
        self.assertEqual(cheap.phenotype.connections[0].cost, 20)
        self.assertEqual(expensive.phenotype.connections[0].cost, 30)
        ind = genome()
        ind.mode_pref[0] = 3
        self.assertEqual(self.evaluate(ind).phenotype.connections[0].mode, "train+bus")

    def test_no_connections_is_explicit_infeasibility_not_fake_fare(self):
        ind = self.evaluate()
        self.assertEqual(ind.phenotype.connections, [])
        self.assertGreater(ind.constraint_violation, 0)
        self.assertTrue(ind.violations)

    def test_sleep_overlap_includes_daytime_departure_and_22_boundary(self):
        day_to_night = connection(0, 1, "2026-12-10T09:50+01:00", "2026-12-11T05:25+00:00")
        leave, arrive = travel_window(day_to_night)
        self.assertAlmostEqual(sleep_overlap(leave, arrive, "Stuttgart"), 8 + 25 / 60)
        overnight = connection(0, 1, "2026-12-10T22:00+01:00", "2026-12-11T05:00+00:00")
        leave, arrive = travel_window(overnight)
        self.assertEqual(sleep_overlap(leave, arrive, "Stuttgart"), 8)

    def test_overnight_discomfort_and_transfers_are_accounted_for(self):
        day = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T16:00+00:00")
        night = connection(0, 1, "2026-12-10T22:00+01:00", "2026-12-11T05:00+00:00")
        day_value = objectives.objective_values(Phenotype([0, 1], [], [day], []))[1]
        night_value = objectives.objective_values(Phenotype([0, 1], [], [night], []))[1]
        leave, arrive = travel_window(night)
        expected = (8 * config.SLEEP_DISRUPTION_PER_HOUR
                    + config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR[night.mode] * clock_window_overlap(leave, arrive, "Stuttgart", 1, 5)
                    + config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR[night.mode] * clock_window_overlap(leave, arrive, "Stuttgart", 9, 20)
                    - config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR[night.mode] * clock_window_overlap(*travel_window(day), "Stuttgart", 9, 20)
                    + config.TRAVEL_DISCOMFORT_PER_HOUR[night.mode] * (elapsed_hours(leave, arrive) - elapsed_hours(*travel_window(day))))
        self.assertAlmostEqual(night_value - day_value, expected)
        changed = objectives.objective_values(Phenotype([0, 1], [], [replace(day, stops=2)], []))[1]
        self.assertEqual(changed - day_value, 2 * config.TRANSFER_DISCOMFORT)

    def test_hotel_wait_discomfort_uses_provider_times(self):
        from objectives import hotel_wait_discomfort
        from models import HotelStay, Visit
        tz = data_loader.city_zone("London")
        def visit(arr, dep):
            return Visit(1, datetime.fromisoformat(arr).replace(tzinfo=tz),
                         datetime.fromisoformat(dep).replace(tzinfo=tz), 1, 1, 9)
        stay = HotelStay("H", 2, 0, 1, 50, 10, 3, city=1, check_in_hour=15.0, check_out_hour=11.0)
        free = visit("2026-12-10T12:00", "2026-12-11T12:00")
        self.assertEqual(hotel_wait_discomfort(free, stay), 0)
        early = visit("2026-12-10T08:00", "2026-12-11T11:30")
        self.assertAlmostEqual(hotel_wait_discomfort(early, stay), 4 * config.CHECK_IN_WAIT_DISCOMFORT_PER_HOUR)
        late = visit("2026-12-10T12:00", "2026-12-11T20:00")
        self.assertEqual(hotel_wait_discomfort(late, stay), config.CHECK_OUT_WAIT_DISCOMFORT)
        self.assertEqual(data_loader.parse_clock_hour("2:00\u202fPM"), 14.0)
        self.assertEqual(data_loader.parse_clock_hour("12:00 AM"), 0.0)
        self.assertIsNone(data_loader.parse_clock_hour("noon"))

    def test_night_and_daytime_penalties_and_mode_order(self):
        for mode in ("train", "bus", "train+bus", "flight"):
            self.assertGreater(config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR[mode],
                               config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR[mode])
        self.assertGreater(config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR["bus"], config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR["train"])
        self.assertGreater(config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR["bus"], config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR["train"])
        start = datetime.fromisoformat("2026-12-10T08:00+01:00")
        end = datetime.fromisoformat("2026-12-10T21:00+01:00")
        self.assertEqual(clock_window_overlap(start, end, "Stuttgart", 9, 20), 11)
        start = datetime.fromisoformat("2026-12-11T00:00+01:00")
        end = datetime.fromisoformat("2026-12-11T06:00+01:00")
        self.assertEqual(clock_window_overlap(start, end, "Stuttgart", 1, 5), 4)

    def test_buffer_time_affects_duration_and_earliest_departure(self):
        flight = replace(self.outbound, mode="flight", access_hours=3, egress_hours=2)
        leave, arrive = travel_window(flight)
        self.assertEqual(elapsed_hours(leave, arrive), flight.duration + 5)
        too_early = connection(0, 1, "2026-12-10T01:00+01:00", "2026-12-10T03:00+00:00",
                               mode="flight", access_hours=3)
        self.install(too_early, self.inbound)
        self.assertGreater(self.evaluate().constraint_violation, 0)

    def test_dst_elapsed_time_uses_utc(self):
        zone = ZoneInfo("Europe/Berlin")
        start = datetime(2026, 10, 25, 1, tzinfo=zone)
        end = datetime(2026, 10, 25, 4, tzinfo=zone)
        self.assertEqual(elapsed_hours(start, end), 4)
        self.assertEqual(sleep_overlap(start, end, "Stuttgart"), 4)

    def test_normalized_budget_and_carbon_violations(self):
        self.install(replace(self.outbound, cost=180, co2=1200), replace(self.inbound, cost=180, co2=1200))
        ind = self.evaluate()
        self.assertAlmostEqual(ind.constraint_violation, 0.4)
        self.assertEqual(set(ind.violations), {"Budget exceeded", "Climate-impact allowance exceeded"})

    def test_verifier_rejects_tampered_stays_and_connections(self):
        self.install(self.outbound, self.inbound)
        ind = self.evaluate()
        self.assertEqual(objectives.check_solution(ind.phenotype), [])
        ind.phenotype.visits[0].nights = 99
        self.assertTrue(any("Incorrect stay" in e for e in objectives.check_solution(ind.phenotype)))
        ind.phenotype.connections[1] = replace(self.inbound, src=2)
        self.assertTrue(any("Disconnected" in e for e in objectives.check_solution(ind.phenotype)))

    def test_hotel_uses_arrival_nights_and_exposes_carbon_alternatives(self):
        self.stack.enter_context(patch.object(config, "TOTAL_DAYS", 4))
        overnight = connection(0, 1, "2026-12-10T09:50+01:00", "2026-12-11T05:25+00:00")
        onward = replace(self.inbound, day=3,
                          departure_time=self.inbound.departure_time + timedelta(days=1),
                          arrival_time=self.inbound.arrival_time + timedelta(days=1))
        self.install(overnight, onward)
        nsga2.HOTELS_DB.extend([
            Hotel(1, 1, {1: 30, 2: 40}, 10, 20, hotel_name="A"),
            Hotel(1, 1, {1: 40, 2: 50}, 10, 1, hotel_name="B")])
        with patch.object(config, "OPTIMIZE_HOTELS", True):
            population = [replace(genome(), hotel_choice=[key]) for key in (0.0, 1.0)]
            nsga2.evaluate_population(population)
            self.assertEqual(len(population), 2)
            ind = next(p for p in population if p.phenotype.hotels[0].hotel_name == "B")
            self.assertEqual(ind.constraint_violation, 0, ind.violations)
            hotel = ind.phenotype.hotels[0]
            self.assertEqual((hotel.check_in_day, hotel.nights, hotel.hotel_name), (1, 2, "B"))
            self.assertEqual(hotel.total_cost, 90)
            self.assertEqual(hotel.total_co2, 2)
            nsga2.HOTELS_DB.clear()
            self.assertGreater(self.evaluate(ind).constraint_violation, 0)
            self.assertEqual(ind.phenotype.hotels, [])

    def test_hotel_loading_is_opt_in_and_does_not_advance_search_rng(self):
        random.seed(12)
        before = random.getstate()
        self.assertEqual(data_loader.load_hotels(), [])
        self.assertEqual(before, random.getstate())

    def test_same_named_hotels_keep_identity_through_decoding_and_validation(self):
        self.install(self.outbound, self.inbound)
        nsga2.HOTELS_DB.extend([
            Hotel(1, 2, {0: 30, 1: 40}, 3, 15, hotel_name="Example", property_token="a"),
            Hotel(1, 2, {0: 40, 1: 50}, 3, 10, hotel_name="Example", property_token="b"),
        ])
        with patch.object(config, "OPTIMIZE_HOTELS", True):
            population = [replace(genome(), hotel_choice=[key]) for key in (0.0, 1.0)]
            nsga2.evaluate_population(population)
            self.assertEqual(len(population), 2)
            ind = next(p for p in population if p.phenotype.hotels[0].property_token == "b")
            self.assertEqual(ind.constraint_violation, 0, ind.violations)
            stay = ind.phenotype.hotels[0]
            self.assertEqual((stay.property_token, stay.total_cost, stay.total_co2), ("b", 90, 20))
            stay.property_token = "a"
            self.assertTrue(any("Incorrect hotel" in error for error in objectives.check_solution(ind.phenotype)))

    def test_recommendations_tie_break_and_do_not_renormalize_to_front_extremes(self):
        a, b = genome(), genome()
        a.objectives, b.objectives = (100, 200, 50), (90, 210, 50)
        options = main.select_best_options([a, b], objectives.ObjectiveBalance((200, 100, 100), (0.5, 0.25, 0.25)))
        self.assertEqual(list(options), ["Cost Optimum", "Comfort Optimum", "Ecological Optimum", "Balanced Optimum"])
        self.assertIs(options["Ecological Optimum"], b)
        self.assertIs(options["Cost Optimum"], b)
        self.assertIs(options["Comfort Optimum"], a)
        self.install(self.outbound, self.inbound)
        ind = self.evaluate()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main.print_itinerary("Example", ind)
        self.assertIn("Total cost (transfer only)", output.getvalue())
        self.assertNotIn("Excluded:", output.getvalue())
        self.assertNotIn("NOT verified transfers", output.getvalue())

    def test_clone_does_not_retain_evaluation_or_population_graph(self):
        ind = genome()
        ind.phenotype = Phenotype([0, 1, 0], [], [], [])
        ind.dominated_set = [ind]
        child = nsga2.clone(ind)
        self.assertIsNone(child.phenotype)
        self.assertFalse(hasattr(child, "dominated_set"))

    def test_itinerary_table_contains_every_segment_including_return(self):
        legs = [
            connection(0, 2, "2026-12-10T05:30+01:00", "2026-12-10T13:55+01:00",
                       cost=30.98, co2=16.88, station_src="Stuttgart Airport",
                       station_dest="Paris (Bercy Seine)", fare_source="flixbus",
                       quoted_at="2026-10-30T23:15:26+00:00", access_hours=0.5, egress_hours=0.5),
            connection(2, 1, "2026-12-12T22:15+01:00", "2026-12-13T05:45+00:00",
                       mode="train+bus", cost=30.98, co2=11.59, stops=1),
            connection(1, 0, "2026-12-14T19:50+00:00", "2026-12-14T22:15+01:00",
                       mode="flight", cost=42, co2=85, station_src="STN", station_dest="FKB",
                       access_hours=4, egress_hours=3),
        ]
        legs[0] = replace(legs[0], provider="FlixBus CEE South d.o.o.")
        ind = genome((2, 1), (2, 2))
        ind.phenotype = Phenotype([0, 2, 1, 0], [2, 2], legs, [])
        ind.objectives = objectives.objective_values(ind.phenotype)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(main, "get_terminal_size", return_value=os.terminal_size((240, 24))):
            main.print_itinerary("Example", ind)
        text = output.getvalue()
        table = [line for line in text.splitlines() if line.startswith("|")]
        self.assertEqual(len(table), 4)
        cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in table]
        self.assertEqual(cells[0], ["Leg", "From", "To", "Departs", "Arrives",
                                    "Duration", "Mode", "Operator", "Fare (EUR)", "CO2 (kg)"])
        for i, (row, leg) in enumerate(zip(cells[1:], legs), start=1):
            self.assertEqual(row, [
                str(i), leg.src_name, leg.dest_name,
                leg.departure_time.isoformat(sep=" ", timespec="minutes"),
                leg.arrival_time.isoformat(sep=" ", timespec="minutes"),
                ["8h 25m", "8h 30m", "1h 25m"][i - 1], leg.mode.upper(), leg.provider,
                f"{leg.cost:.2f}", f"{leg.co2:.2f}",
            ])
        self.assertEqual(len({len(line) for line in table}), 1)
        self.assertIn("STN -> FKB", text)
        self.assertIn("With allowances: 8h 25m", text)
        self.assertNotIn("Fare quote:", text)
        self.assertNotIn("2026-10-30T23:15:26+00:00", text)
        for removed in ("Modeled travel window:", "Elapsed:", "Return before",
                        "Times are scheduled", "CO2 is the base", "Excluded:",
                        "Access/egress times", "CO2 method:"):
            self.assertNotIn(removed, text)
        self.assertNotIn("Stay:", text)
        self.assertNotIn("Sleep-window travel:", text)
        self.assertNotIn("Usable", text)

    def test_itinerary_table_wraps_without_losing_cells(self):
        rows = [
            ("1", "Stuttgart", "Paris", "2026-12-10 05:30+01:00", "2026-12-10 13:55+01:00",
             "8h 25m", "BUS", "FlixBus CEE South d.o.o.", "30.98", "16.88"),
            ("2", "Paris", "Stuttgart", "2026-12-14 19:50+01:00", "2026-12-15 05:45+01:00",
             "9h 55m", "TRAIN+BUS", "A very long operator name without truncation", "123.45", "11.59"),
            ("3", "Paris", "Paris", "2026-12-15", "2026-12-17", "2 nights", "HOTEL",
             "A very long hotel name without truncation [SYNTHETIC]", "125.67", "31.00"),
        ]
        for width in (80, 100, 120, 240):
            with self.subTest(width=width):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), \
                        patch.object(main, "get_terminal_size", return_value=os.terminal_size((width, 24))):
                    main.print_itinerary_table(rows, include_hotels=True)
                lines = output.getvalue().splitlines()
                self.assertLessEqual(max(map(len, lines)), width)
                self.assertEqual(len({len(line) for line in lines}), 1)
                self.assertTrue(lines[0].startswith("+"))
                groups, current = [], []
                for line in lines[1:]:
                    if line.startswith("+"):
                        groups.append(current)
                        current = []
                    else:
                        current.append([cell.strip() for cell in line[1:-1].split("|")])
                self.assertEqual(len(groups), len(rows) + 1)
                for group, expected in zip(groups[1:], rows):
                    for col, value in enumerate(expected):
                        reconstructed = "".join(line[col] for line in group)
                        self.assertEqual("".join(reconstructed.split()), "".join(value.split()))

    def test_itinerary_interleaves_hotels_with_dates_and_stay_totals(self):
        ind = genome()
        stay = HotelStay("Example hotel", 1, 0, 2, 90, 2, 20, city=1, synthetic=True)
        ind.phenotype = Phenotype([0, 1, 0], [2], [self.outbound, self.inbound], [stay])
        output = io.StringIO()
        with patch.object(config, "OPTIMIZE_HOTELS", True), contextlib.redirect_stdout(output), \
                patch.object(main, "get_terminal_size", return_value=os.terminal_size((300, 24))):
            ind.objectives = objectives.objective_values(ind.phenotype)
            main.print_itinerary("Example", ind)
        text = output.getvalue()
        self.assertIn("Total Cost (transfer + accommodation): EUR 130.00", text)
        table = [line for line in text.splitlines() if line.startswith("|")]
        cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in table]
        self.assertEqual(cells[0], ["Step", "From", "To", "Departs / Check-in", "Arrives / Check-out",
                                    "Duration", "Type", "Operator / Hotel", "Cost (EUR)", "CO2 (kg)"])
        self.assertEqual([row[6] for row in cells[1:]], ["BUS", "HOTEL", "BUS"])
        self.assertEqual(cells[2], ["2", "London", "London", "2026-12-10", "2026-12-12",
                                    "2 nights", "HOTEL", "Example hotel", "90.00", "2.00"])
        self.assertEqual(cells[3][0], "3")
        self.assertIn("  1. Stuttgart -> London", text)
        self.assertIn("  3. London -> Stuttgart", text)
        self.assertNotIn("  2. London -> Stuttgart", text)
        self.assertNotIn("Accommodation (not included", text)

    def test_hotel_rows_follow_overnight_arrivals_and_match_cities_not_list_order(self):
        legs = [
            connection(0, 2, "2026-12-10T09:00+01:00", "2026-12-10T18:00+01:00"),
            connection(2, 1, "2026-12-12T21:30+01:00", "2026-12-13T05:00+00:00"),
            connection(1, 0, "2026-12-15T10:00+00:00", "2026-12-15T17:00+01:00"),
        ]
        stays = [
            HotelStay("London hotel", 2, 3, 2, 100, 31, 6, city=1),
            HotelStay("Paris hotel", 2, 0, 2, 110, 31, 6, city=2),
        ]
        ind = genome((2, 1), (2, 2))
        ind.phenotype = Phenotype([0, 2, 1, 0], [2, 2], legs, stays)
        with patch.object(config, "OPTIMIZE_HOTELS", True), \
                patch.object(main, "print_itinerary_table") as table, \
                contextlib.redirect_stdout(io.StringIO()):
            ind.objectives = objectives.objective_values(ind.phenotype)
            main.print_itinerary("Example", ind)
        rows = table.call_args.args[0]
        self.assertEqual([row[0] for row in rows], ["1", "2", "3", "4", "5"])
        self.assertEqual([row[6] for row in rows], ["BUS", "HOTEL", "BUS", "HOTEL", "BUS"])
        self.assertEqual(rows[1][7], "Paris hotel")
        self.assertEqual(rows[3][3:5], ("2026-12-13", "2026-12-15"))
        self.assertEqual(rows[3][7], "London hotel")
        self.assertEqual(rows[-1][2], "Stuttgart")
        with patch.object(config, "OPTIMIZE_HOTELS", False), \
                patch.object(main, "print_itinerary_table") as table, \
                contextlib.redirect_stdout(io.StringIO()):
            main.print_itinerary("Transport only", ind)
        self.assertEqual(len(table.call_args.args[0]), 3)
        self.assertFalse(table.call_args.kwargs["include_hotels"])

    def test_duration_display_rounds_to_minutes_and_handles_long_journeys(self):
        self.assertEqual(main.format_duration(8 + 25 / 60), "8h 25m")
        self.assertEqual(main.format_duration(1.999), "2h 00m")
        self.assertEqual(main.format_duration(27.5), "27h 30m")

    def test_route_variation_keeps_city_nights_and_required_cities(self):
        with patch.multiple(config, TOTAL_DAYS=10, MUST_VISIT_DESTINATIONS=["London"]), \
                patch.object(config, "CROSSOVER_RATE", 1.0):
            first = genome((1, 2), (3, 2), (0.1, 0.2, 0.3))
            second = genome((2, 1), (2, 3), (0.8, 0.7, 0.6))
            random.seed(42)
            for _ in range(30):
                children = nsga2.crossover(first, second)
                for child in children:
                    self.assertEqual(dict(zip(child.route, child.nights)), {1: 3, 2: 2})
                    self.assertEqual(len(child.connection_choice), 3)
                    nsga2.mutate(child)
                    self.assertIn(1, child.route)
                    self.assertEqual(len(set(child.route)), len(child.route))
                    self.assertTrue(all(n >= 2 for n in child.nights))

    def test_initial_rank_and_search_match_exhaustive_tiny_front(self):
        cheap_slow = self.outbound
        expensive_fast = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T12:00+00:00", cost=30)
        self.install(cheap_slow, expensive_fast, self.inbound)
        exhaustive = [self.evaluate(genome(choices=(key, 0))) for key in (0, 1)]
        expected = {p.objectives for p in nsga2.rank_population(exhaustive)[0]}
        self.assertEqual(len(expected), 2)
        with patch.multiple(config, TOTAL_DAYS=3, TRAVEL_RANGE=["2026-12-10", "2026-12-12"]), \
                patch.multiple(nsga2, NUM_DESTINATIONS=1, TOTAL_AVAILABLE_DESTINATIONS=1), \
                patch.multiple(main, POPULATION_SIZE=10, MAX_GENERATIONS=5):
            front = main.optimize()
            repeat = main.optimize()
        self.assertEqual({p.objectives for p in front}, expected)
        self.assertEqual([p.objectives for p in front], [p.objectives for p in repeat])
        self.assertTrue(all(p.rank == 0 for p in front))
        self.assertTrue(all(not objectives.check_solution(p.phenotype) for p in front))

    def test_constant_axis_does_not_arbitrarily_assign_infinite_crowding(self):
        front = [genome() for _ in range(4)]
        for ind in front:
            ind.objectives = (10, 10, 10)
        nsga2.crowding_distance_assignment(front)
        self.assertTrue(all(p.crowding_distance == 0 for p in front))

    def test_impossible_configuration_is_rejected_before_fetching(self):
        with patch.object(config, "TOTAL_DAYS", 2), patch.object(main, "initialize_database") as load:
            with self.assertRaisesRegex(ValueError, "too short"):
                main.main()
            load.assert_not_called()

    def test_evaluation_reports_completed_candidates_incrementally_without_changing_results(self):
        self.install(self.outbound, self.inbound)
        baseline = [genome(), genome()]
        nsga2.evaluate_population(baseline)
        batch = [genome(), genome()]
        search = decoder.TransportSearch()
        reports = []

        def report(candidates):
            reports.append((search.generated, [p.objectives for p in candidates]))

        nsga2.evaluate_population(batch, search=search, on_evaluated=report)
        self.assertEqual([count for count, _ in reports], [1, 2])
        self.assertEqual([p.objectives for p in batch], [p.objectives for p in baseline])
        self.assertEqual([values for _, group in reports for values in group],
                         [p.objectives for p in batch])


class ProgressTests(unittest.TestCase):
    def candidate(self, values, identifier="a", violation=0, reasons=()):
        ind = genome()
        conn = connection(0, 1, "2026-12-10T09:00+01:00", "2026-12-10T17:00+00:00",
                          journey_id=identifier)
        ind.phenotype = Phenotype([0, 1, 0], [2], [conn], [])
        ind.objectives = values
        ind.constraint_violation = violation
        ind.violations = list(reasons)
        return ind

    def report(self, pop, previous=None, width=120):
        front = nsga2.rank_population(pop)[0]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(main, "get_terminal_size", return_value=os.terminal_size((width, 24))):
            points = main.print_optimization_progress(pop, front, 25, 52.5, previous_points=previous)
        return output.getvalue(), points

    def test_reports_all_axes_from_front_not_whole_population(self):
        cheap = self.candidate((100, 80, 60), "cheap")
        comfortable = self.candidate((150, 40, 20), "comfortable")
        dominated = self.candidate((200, 90, 90), "dominated")
        invalid = self.candidate((1, 1, 1), "invalid", violation=1, reasons=["Budget exceeded"])
        previous = frozenset({(100, 90, 70), (150, 50, 25)})
        text, points = self.report([cheap, comfortable, dominated, invalid, cheap], previous)
        self.assertEqual(points, frozenset({cheap.objectives, comfortable.objectives}))
        self.assertRegex(text, r"100\.00\s+\+0\.00\s+20\.0\s+-5\.0\s+40\.0\s+-10\.0")
        self.assertIn("025/", text)
        self.assertIn("00:52", text)
        self.assertEqual(len(text.splitlines()), 3)
        for removed in ("feasible", "diversity", "hypervolume", "different plans", "violation"):
            self.assertNotIn(removed, text.lower())

    def test_equal_objective_points_and_unchanged_values(self):
        a = self.candidate((100, 50, 20), "a")
        b = self.candidate((100, 50, 20), "b")
        text, points = self.report([a, a, b])
        self.assertEqual(points, frozenset({a.objectives}))
        self.assertRegex(text, r"100\.00\s+--\s+20\.0\s+--\s+50\.0\s+--")
        repeat, _ = self.report([a, b], points)
        self.assertRegex(repeat, r"100\.00\s+\+0\.00\s+20\.0\s+\+0\.0\s+50\.0\s+\+0\.0")

    def test_reports_worsening_as_positive_not_as_improvement(self):
        ind = self.candidate((110, 50, 20))
        text, _ = self.report([ind], frozenset({(100, 40, 15)}))
        self.assertRegex(text, r"110\.00\s+\+10\.00\s+20\.0\s+\+5\.0\s+50\.0\s+\+10\.0")

    def test_no_feasible_front_shows_placeholders_instead_of_partial_costs(self):
        a = self.candidate((10, 20, 30), violation=0.4, reasons=["Budget exceeded", "Budget exceeded"])
        b = self.candidate((1, 2, 3), violation=2, reasons=["Budget exceeded", "Incomplete transport itinerary"])
        text, points = self.report([a, b])
        self.assertFalse(points)
        self.assertRegex(text.splitlines()[-1], r"(?:--\s+){5}--$")
        self.assertNotIn("Budget exceeded", text)
        self.assertNotIn("violation", text)
        self.assertNotIn("10.00", text)

    def test_progress_fits_narrow_terminals_without_mutating_candidates_or_rng(self):
        ind = self.candidate((123456.78, 2345.6, 3456.7))
        before = random.getstate()
        for width in (40, 80, 120):
            with self.subTest(width=width):
                text, _ = self.report([ind], width=width)
                self.assertLessEqual(max(map(len, text.splitlines())), width)
                compact = "".join(text.split())
                for value in ("123456.78", "2345.6", "3456.7", "Discomfort"):
                    self.assertIn(value, compact)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(ind.objectives, (123456.78, 2345.6, 3456.7))

    def test_initial_interval_and_terminal_reports_for_all_stop_conditions(self):
        ind = self.candidate((100, 50, 20))
        for reason in ("generation", "runtime", "stagnation"):
            with self.subTest(reason=reason), contextlib.ExitStack() as stack:
                stack.enter_context(patch.multiple(main, POPULATION_SIZE=2, MAX_GENERATIONS=26))
                stack.enter_context(patch.multiple(
                    config, MAX_RUNTIME_SECONDS=10 if reason == "runtime" else 1000,
                    # Local bombs deliberately postpone a plateau stop; this case
                    # covers the plain stopping contract, not the two-stage search.
                    BOMB_STRENGTH=0,
                    QUALITY_WINDOW_GENERATIONS=2 if reason == "stagnation" else 100,
                    QUALITY_HV_MIN_GAIN=0 if reason == "stagnation" else None,
                    QUALITY_COMPROMISE_MIN_GAIN=0 if reason == "stagnation" else None))
                stack.enter_context(patch.object(main, "initialize_population", return_value=[ind, ind]))
                stack.enter_context(patch.object(main, "evaluate_population"))
                stack.enter_context(patch.object(main, "tournament_selection", return_value=ind))
                stack.enter_context(patch.object(main, "crossover", return_value=(ind, ind)))
                stack.enter_context(patch.object(main, "mutate"))
                stack.enter_context(patch.object(main, "select_survivors", return_value=[ind, ind]))
                stack.enter_context(patch.object(main, "check_solution", return_value=[]))
                stack.enter_context(patch.object(main, "objective_values", return_value=ind.objectives))
                clock = [0.0]
                stack.enter_context(patch.object(main, "monotonic", side_effect=lambda: clock[0]))
                def advance(*args):
                    clock[0] += 1.0
                stack.enter_context(patch.object(main, "evaluate_population", side_effect=advance))
                stack.enter_context(patch.object(decoder.TransportSearch, "expired", return_value=False))
                report = stack.enter_context(patch.object(main, "print_optimization_progress", return_value=frozenset()))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                main.optimize(balance=objectives.ObjectiveBalance((200, 100, 100), (0.5, 0.25, 0.25)))
                generations = [call.args[2] for call in report.call_args_list]
                self.assertEqual(generations[:3], [0, 0, 1])
                self.assertEqual(generations[-1], {"generation": 26, "runtime": 10, "stagnation": 2}[reason])
                self.assertEqual(report.call_args_list[-1].kwargs["phase"], "Done")
                if reason != "stagnation":
                    self.assertIn(5, generations)
                self.assertEqual(sum(call.kwargs["show_header"] for call in report.call_args_list), 1)
                message = {"generation": "generation limit", "runtime": "runtime limit",
                           "stagnation": "quality plateau"}[reason]
                self.assertIn(message, output.getvalue())

    def test_reports_during_initialization_before_generation_zero_completes(self):
        first = self.candidate((100, 50, 20))
        better = self.candidate((90, 40, 15), "better")
        clock = [0.0]

        def initialize(size, hotels, search, on_evaluated):
            clock[0] = 6.0
            on_evaluated([first])
            clock[0] = 12.0
            on_evaluated([better])
            clock[0] = 20.0
            return [first, better]

        output = io.StringIO()
        with patch.multiple(main, POPULATION_SIZE=2, MAX_GENERATIONS=3), \
                patch.object(config, "MAX_RUNTIME_SECONDS", 20), \
                patch.object(main, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(main, "initialize_population", side_effect=initialize), \
                patch.object(main, "select_survivors", side_effect=lambda pop, *args: pop), \
                patch.object(decoder.TransportSearch, "expired", side_effect=lambda: clock[0] >= 20), \
                patch.object(main, "check_solution", return_value=[]), \
                patch.object(main, "objective_values", side_effect=lambda p: better.objectives), \
                patch.object(main, "get_terminal_size", return_value=os.terminal_size((120, 24))), \
                contextlib.redirect_stdout(output):
            front = main.optimize(balance=objectives.ObjectiveBalance((200, 100, 100), (0.5, 0.25, 0.25)))
        text = output.getvalue()
        self.assertRegex(text, r"Init\s+000/003\s+00:06\s+100\.00")
        self.assertRegex(text, r"Init\s+000/003\s+00:12\s+90\.00\s+-10\.00")
        self.assertRegex(text, r"Done\s+000/003\s+00:20\s+90\.00")
        self.assertIn("runtime limit during initialization", text)
        self.assertEqual(front, [better])


if __name__ == "__main__":
    unittest.main()
