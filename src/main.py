import sys
from pathlib import Path
for _sub in ('', 'helpers', 'optimizer'):
    sys.path.insert(0, str(Path(__file__).resolve().parent / _sub))
import argparse
from collections import Counter
from datetime import date, timedelta
import math
import random
from shutil import get_terminal_size
from textwrap import wrap
from time import monotonic

import config
from accommodation import HotelFrontiers
from config import POPULATION_SIZE, MAX_GENERATIONS
from data_loader import CITIES, HOTELS_DB, initialize_database, print_database_summary
from decoder import TransportSearch
from models import Chromosome
from nsga2 import (initialize_population, evaluate_population, tournament_selection,
                                 crossover, mutate, select_survivors, unique_individuals, parent_groups,
                                 complementary_partner)
from objectives import ObjectiveBalance, check_solution, objective_dominates, objective_values
from progress import MutationSchedule, OffspringStats, ParetoArchive, QualityMonitor
from timing import departure_days, elapsed_hours, minimum_nights, travel_window, trip_bounds


def validate_config():
    if (isinstance(config.NUM_TRAVELLERS, bool) or not isinstance(config.NUM_TRAVELLERS, int)
            or config.NUM_TRAVELLERS < 1):
        raise ValueError("NUM_TRAVELLERS must be a positive integer")
    if (not isinstance(config.TOTAL_DAYS, int) or config.TOTAL_DAYS < 1
            or not isinstance(config.NUM_DESTINATIONS, int)):
        raise ValueError("Trip days and destination count must be positive integers")
    if (config.NUM_DESTINATIONS < 1 or config.NUM_DESTINATIONS > len(config.DESTINATIONS)
            or len(config.DESTINATIONS) != len(set(config.DESTINATIONS))
            or config.START_CITY in config.DESTINATIONS):
        raise ValueError("Choose distinct destinations and a valid destination count")
    if (len(set(config.MUST_VISIT_DESTINATIONS)) != len(config.MUST_VISIT_DESTINATIONS)
            or not set(config.MUST_VISIT_DESTINATIONS).issubset(config.DESTINATIONS)
            or len(config.MUST_VISIT_DESTINATIONS) > config.NUM_DESTINATIONS):
        raise ValueError("MUST_VISIT_DESTINATIONS must be a distinct subset fitting NUM_DESTINATIONS")
    minima = {city: minimum_nights(city) for city in config.DESTINATIONS}
    if any(not isinstance(n, int) or n < 1 for n in minima.values()):
        raise ValueError("Minimum destination nights must be positive integers")
    if not set(config.MIN_NIGHTS_BY_CITY).issubset(config.DESTINATIONS):
        raise ValueError("Minimum-night overrides must name configured destinations")
    for city in [config.START_CITY, *config.DESTINATIONS]:
        if city not in config.CITY_TIMEZONES:
            raise ValueError(f"Missing city timezone: {city}")
    required = sum(minima[c] for c in config.MUST_VISIT_DESTINATIONS)
    optional = sorted(n for c, n in minima.items() if c not in config.MUST_VISIT_DESTINATIONS)
    minimum_total = required + sum(optional[:config.NUM_DESTINATIONS - len(config.MUST_VISIT_DESTINATIONS)])
    start, end = trip_bounds()
    calendar_days = (end.date() - start.date()).days
    if calendar_days <= minimum_total:
        raise ValueError("Travel window is too short for the required destination nights")
    if type(POPULATION_SIZE) is not int or POPULATION_SIZE < 2:
        raise ValueError("POPULATION_SIZE must be an integer >= 2")
    if type(MAX_GENERATIONS) is not int or MAX_GENERATIONS < 1:
        raise ValueError("MAX_GENERATIONS must be a positive integer")
    if not math.isfinite(config.MAX_RUNTIME_SECONDS) or config.MAX_RUNTIME_SECONDS <= 0:
        raise ValueError("MAX_RUNTIME_SECONDS must be finite and positive")
    if type(config.RANDOM_SEED) is not int:
        raise ValueError("RANDOM_SEED must be an integer")
    QualityMonitor()
    nonnegative = (
        config.MIN_USABLE_HOURS_PER_CITY, config.STATION_ACCESS_HOURS, config.STATION_EXIT_HOURS,
        config.FLIGHT_CHECK_IN_HOURS, config.FLIGHT_EXIT_HOURS, *config.AIRPORT_ACCESS_HOURS.values(),
        *config.TRAVEL_DISCOMFORT_PER_HOUR.values(), config.FLIGHT_DISCOMFORT,
        *config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR.values(),
        *config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR.values(),
        config.TRANSFER_DISCOMFORT, config.SLEEP_DISRUPTION_PER_HOUR,
        config.LONG_JOURNEY_THRESHOLD_HOURS, config.LONG_JOURNEY_DISCOMFORT_PER_HOUR,
        config.CHECK_IN_FREE_HOURS, config.CHECK_IN_WAIT_DISCOMFORT_PER_HOUR,
        config.CHECK_OUT_FREE_HOURS, config.CHECK_OUT_WAIT_DISCOMFORT)
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("Timing allowances and discomfort weights must be finite and non-negative")
    if (not 0 <= config.DAYTIME_START_HOUR < config.DAYTIME_END_HOUR <= 24
            or not 0 <= config.NIGHT_START_HOUR < config.NIGHT_END_HOUR <= 24
            or set(config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR) != set(config.TRAVEL_DISCOMFORT_PER_HOUR)
            or set(config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR) != set(config.TRAVEL_DISCOMFORT_PER_HOUR)):
        raise ValueError("Daytime/night windows must be valid and define a weight for every mode")
    if (not 0 <= config.SLEEP_END_HOUR < config.SLEEP_START_HOUR < 24
            or not 0 < config.MIN_REST_HOURS <= 24 - config.SLEEP_START_HOUR + config.SLEEP_END_HOUR):
        raise ValueError("Rest hours must fit the configured overnight sleep window")
    if not (0 <= config.MUTATION_RATE <= 1 and 0 <= config.CROSSOVER_RATE <= 1):
        raise ValueError("Mutation and crossover rates must be in [0, 1]")
    adaptive = (config.MUTATION_RATE_MIN, config.MUTATION_RATE_MAX, config.STEP_GROWTH,
                config.SUCCESS_TARGET)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
           for value in adaptive):
        raise ValueError("Adaptive mutation settings must be finite numbers")
    if not 0 < config.MUTATION_RATE_MIN <= config.MUTATION_RATE_MAX <= 1:
        raise ValueError("MUTATION_RATE bounds must satisfy 0 < MIN <= MAX <= 1")
    if not 0 < config.SUCCESS_TARGET < 1:
        raise ValueError("SUCCESS_TARGET must lie strictly between 0 and 1")
    if config.STEP_GROWTH <= 1:
        raise ValueError("STEP_GROWTH must be greater than 1")
    if (type(config.TRANSPORT_CACHE_SIZE) is not int or config.TRANSPORT_CACHE_SIZE < 0
            or type(config.TRANSPORT_RESULT_CACHE_SIZE) is not int or config.TRANSPORT_RESULT_CACHE_SIZE < 0):
        raise ValueError("Cache limits must be non-negative integers")
    if not 0 <= config.COORDINATED_MUTATION_RATE <= 1:
        raise ValueError("Coordinated mutation rate must be in [0, 1]")
    if any(type(value) is not bool for value in (
            config.PRUNE_EQUIVALENT_TRANSPORT, config.TRANSPORT_GROUP_SELECTION,
            config.COMPLEMENTARY_PAIRING)):
        raise ValueError("Transport pruning, group selection and complementary pairing must be booleans")
    if any(not math.isfinite(value) or value <= 0 for value in (
            config.MAX_BUDGET, config.MAX_CO2, config.MAX_TRANSFER_HOURS)):
        raise ValueError("Budget and carbon/journey limits must be finite and positive")


def select_best_options(pareto_front: list[Chromosome], balance: ObjectiveBalance) -> dict[str, Chromosome]:
    if not pareto_front:
        raise ValueError("Cannot recommend from an empty Pareto front")
    def compromise(ind):
        return balance.score(ind.objectives), ind.objectives
    return {
        "Cost Optimum": min(pareto_front, key=lambda p: (p.objectives[0], p.objectives[2], p.objectives[1])),
        "Comfort Optimum": min(pareto_front, key=lambda p: (p.objectives[1], p.objectives[0], p.objectives[2])),
        "Ecological Optimum": min(pareto_front, key=lambda p: (p.objectives[2], p.objectives[0], p.objectives[1])),
        "Balanced Optimum": min(pareto_front, key=compromise),
    }


def format_duration(hours):
    hours, minutes = divmod(round(hours * 60), 60)
    return f"{hours}h {minutes:02d}m"


def print_itinerary_table(rows, *, include_hotels=False):
    headers = (
        ("Step", "From", "To", "Departs / Check-in", "Arrives / Check-out",
         "Duration", "Type", "Operator / Hotel", "Cost (EUR)", "CO2 (kg)")
        if include_hotels else
        ("Leg", "From", "To", "Departs", "Arrives",
         "Duration", "Mode", "Operator", "Fare (EUR)", "CO2 (kg)")
    )
    widths = [max(len(row[col]) for row in [headers, *rows]) for col in range(len(headers))]
    minimums = (3, 4, 4, 11, 11, 8, 5, 6, 6, 6)
    available = max(41, get_terminal_size(fallback=(120, 24)).columns)
    # Keep dates and times intact where possible; fold rather than omit data on narrow terminals.
    while sum(widths) + 3 * len(widths) + 1 > available:
        candidates = [col for col, width in enumerate(widths) if width > minimums[col]]
        if not candidates:
            candidates = [col for col, width in enumerate(widths) if width > 1]
        col = max(candidates, key=lambda col: widths[col])
        widths[col] -= 1

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    for row_index, row in enumerate([headers, *rows]):
        cells = [wrap(cell, width=width, break_on_hyphens=False) or [""]
                 for cell, width in zip(row, widths)]
        for line_index in range(max(map(len, cells))):
            values = [cell[line_index] if line_index < len(cell) else "" for cell in cells]
            print("| " + " | ".join(
                value.rjust(width) if row_index and col in (0, 8, 9) else value.ljust(width)
                for col, (value, width) in enumerate(zip(values, widths))) + " |")
        print(border.replace("-", "=") if row_index == 0 else border)


def print_itinerary(title, ind):
    cost, discomfort, carbon = ind.objectives
    p = ind.phenotype
    print(f"\n{title}")
    print("=" * min(len(title), get_terminal_size(fallback=(120, 24)).columns))
    label = "Total Cost (transfer + accommodation)" if config.OPTIMIZE_HOTELS else "Total cost (transfer only)"
    print(f"{label}: EUR {cost:.2f}")
    print(f"Discomfort: {discomfort:.1f} | CO2: {carbon:.1f} kg")
    start, deadline = trip_bounds(p.start_day)
    print(f"Trip dates: {start.date()} to {(deadline - timedelta(days=1)).date()}"
          f" ({config.TOTAL_DAYS} calendar days)")

    rows = []
    travel_rows = []
    include_hotels = config.OPTIMIZE_HOTELS and bool(p.hotels)
    hotels_by_city = {stay.city: stay for stay in p.hotels} if include_hotels else {}
    start_date = date.fromisoformat(config.TRAVEL_RANGE[0])
    for conn in p.connections:
        travel_window(conn)
        step = len(rows) + 1
        travel_rows.append((step, conn))
        rows.append((
            str(step), conn.src_name, conn.dest_name,
            conn.departure_time.isoformat(sep=" ", timespec="minutes"),
            conn.arrival_time.isoformat(sep=" ", timespec="minutes"),
            format_duration(conn.duration), conn.mode.upper(), conn.provider,
            f"{conn.cost:.2f}", f"{conn.co2:.2f}",
        ))
        stay = hotels_by_city.get(conn.dest) if conn.dest != 0 else None
        if stay is not None:
            check_in = start_date + timedelta(days=stay.check_in_day)
            check_out = check_in + timedelta(days=stay.nights)
            city = CITIES[stay.city]
            rows.append((
                str(len(rows) + 1), city, city, check_in.isoformat(), check_out.isoformat(),
                f"{stay.nights} night{'s' if stay.nights != 1 else ''}", "HOTEL",
                stay.hotel_name,
                f"{stay.total_cost:.2f}", f"{stay.total_co2:.2f}",
            ))
    print()
    print_itinerary_table(rows, include_hotels=include_hotels)
    print(f"\nAvailable daytime (local {config.DAYTIME_START_HOUR:02d}:00-{config.DAYTIME_END_HOUR:02d}:00):")
    for visit in p.visits:
        print(f"  {CITIES[visit.city]}: {visit.usable_hours:.2f} hours")

    print("\nTravel details:" if include_hotels else "\nLeg details:")
    for i, conn in travel_rows:
        leave, arrive = travel_window(conn)
        print(f"  {i}. {conn.station_src or conn.src_name} -> {conn.station_dest or conn.dest_name}"
              f" | Transfers: {conn.stops} | With allowances: {format_duration(elapsed_hours(leave, arrive))}")


def print_optimization_progress(
    pop: list[Chromosome], front: list[Chromosome], generation: int,
    elapsed: float,
    previous_points: frozenset[tuple[float, float, float]] | None = None,
    *, phase: str = "Search", show_header: bool = True,
) -> frozenset[tuple[float, float, float]]:
    """Print best-so-far objectives and signed changes since the last display."""
    width = max(20, get_terminal_size(fallback=(120, 24)).columns)

    def emit(text):
        print("\n".join(wrap(text, width=width, break_on_hyphens=False)), flush=True)

    pareto = unique_individuals([ind for ind in front if ind.constraint_violation == 0])
    points = frozenset(ind.objectives for ind in pareto)
    values = []
    for axis, precision in ((0, 2), (2, 1), (1, 1)):
        best = min((point[axis] for point in points), default=None)
        change = (round(best - min(point[axis] for point in previous_points), precision)
                  if best is not None and previous_points else None)
        values.append(("--" if best is None else f"{best:.{precision}f}",
                       "--" if change is None else f"{change:+.{precision}f}"))
    minutes, seconds = divmod(int(elapsed), 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    gen = f"{generation:03d}/{MAX_GENERATIONS:03d}"
    headers = ("Stage", "Gen", "Elapsed", "Best EUR", "Change", "CO2 kg", "Change", "Discomfort", "Change")
    row = (phase, gen, clock, *(cell for pair in values for cell in pair))
    widths = (6, 7, 7, 10, 9, 9, 8, 11, 8)

    def table_line(cells):
        return "  ".join(value.ljust(size) if i < 3 else value.rjust(size)
                         for i, (value, size) in enumerate(zip(cells, widths))).rstrip()

    if max(len(table_line(headers)), len(table_line(row))) <= width:
        if show_header:
            emit(table_line(headers))
            emit("-" * len(table_line(headers)))
        emit(table_line(row))
    else:
        emit(f"{phase} | Gen {gen} | {clock}")
        for label, (best, change) in zip(("Best EUR", "CO2 kg", "Discomfort"), values):
            emit(f"  {label}: {best} ({change})")
    return points


def optimize(metrics: dict | None = None, *, balance: ObjectiveBalance | None = None) -> list[Chromosome]:
    balance = ObjectiveBalance.from_data() if balance is None else balance
    random.seed(config.RANDOM_SEED)
    started = monotonic()
    monitor = QualityMonitor(balance)
    archive = ParetoArchive()
    generation = 0
    phase = "Setup"
    reported_points = None
    last_report = -math.inf

    def report_progress(force=False):
        nonlocal reported_points, last_report
        elapsed = monotonic() - started
        if force or elapsed - last_report >= 5:
            reported_points = print_optimization_progress(
                [], archive.front, generation, elapsed, previous_points=reported_points,
                phase=phase, show_header=reported_points is None)
            last_report = elapsed

    def record_candidates(candidates):
        archive.update(candidates)
        report_progress()

    print(f"\nOptimization | {MAX_GENERATIONS} generations | {config.MAX_RUNTIME_SECONDS:g}s"
          " | Change since previous update", flush=True)
    report_progress(force=True)
    hotels = HotelFrontiers(HOTELS_DB) if config.OPTIMIZE_HOTELS else None
    search = TransportSearch(stop_at=started + config.MAX_RUNTIME_SECONDS)
    search.started = started
    setup_seconds = monotonic() - started
    history = []
    observations = []
    schedule = MutationSchedule()
    offspring_stats = None
    def observe_population(generation: int) -> None:
        groups = parent_groups(pop)
        observations.append(dict(
            generation=generation, seconds=monotonic() - started,
            parent_groups=len(groups), largest_group_share=max(map(len, groups)) / len(pop),
            distinct_feasible_transports=len(search.feasible_transports),
            feasible_dates=len({p.start_day for p in pop if not p.constraint_violation}),
            feasible_routes=len({tuple(p.route) for p in pop if not p.constraint_violation}),
            mutation_rate=schedule.rate, mutation_scale=schedule.scale,
            success_ratio=schedule.success_ratio,
            offspring_unique_feasible=offspring_stats.unique_feasible if offspring_stats else None,
            offspring_duplicate_complete=offspring_stats.duplicate_complete if offspring_stats else None,
            objective_minima=[min(p.objectives[i] for p in archive.front) for i in range(3)]
            if archive.front else None, hypervolume=quality.hypervolume, compromise=quality.compromise))
    phase = "Init"
    report_progress(force=True)
    pop = initialize_population(POPULATION_SIZE, hotels, search, record_candidates)
    archive.update(pop)
    pop = select_survivors(pop, POPULATION_SIZE, hotels, record_candidates, search)
    initialization_seconds = monotonic() - started
    initialization_candidates = search.generated
    offspring_candidates = 0
    quality = monitor.observe(archive)
    observe_population(0)
    history.append((0, monotonic() - started, quality.hypervolume, quality.compromise,
                    schedule.rate, schedule.success_ratio))
    monitor.reset_report_counts()
    stop_reason = "generation limit"
    for gen in range(MAX_GENERATIONS):
        if search.expired():
            stop_reason = "runtime limit" + (" during initialization" if generation == 0 else "")
            break
        generation = gen + 1
        if phase != "Search":
            phase = "Search"
            report_progress(force=True)
        offspring = []
        groups = parent_groups(pop) if config.TRANSPORT_GROUP_SELECTION else None
        while len(offspring) < POPULATION_SIZE:
            parent1 = tournament_selection(pop, groups)
            parent2 = (complementary_partner(parent1, pop, balance.scales, groups)
                       if config.COMPLEMENTARY_PAIRING else tournament_selection(pop, groups))
            c1, c2 = crossover(parent1, parent2)
            mutate(c1, schedule)
            mutate(c2, schedule)
            # The first parent is the tournament winner both children descend from.
            c1.parent_objectives = c2.parent_objectives = (
                parent1.objectives if parent1.constraint_violation == 0 else None)
            offspring.extend((c1, c2))
        offspring = offspring[:POPULATION_SIZE]
        before = search.generated
        evaluate_population(offspring, hotels, search, record_candidates)
        offspring_candidates += search.generated - before
        offspring_stats = OffspringStats.measure(offspring, POPULATION_SIZE)
        # Step control counts a Pareto improvement over the parent, the criterion
        # the search is actually ranked by; children of infeasible parents carry
        # no information about step size and stay out of both counts.
        trials = [c for c in offspring if c.parent_objectives is not None]
        successes = sum(1 for c in trials if c.constraint_violation == 0
                        and objective_dominates(c.objectives, c.parent_objectives))
        schedule.observe(successes, len(trials))
        archive.update(offspring)
        pop = select_survivors(pop + offspring, POPULATION_SIZE, hotels, record_candidates, search)
        quality = monitor.observe(archive)
        observe_population(gen + 1)
        elapsed = monotonic() - started
        history.append((gen + 1, elapsed, quality.hypervolume, quality.compromise,
                        schedule.rate, schedule.success_ratio))
        runtime_limit = elapsed >= config.MAX_RUNTIME_SECONDS
        stagnation_limit = quality.stop
        report_progress()
        monitor.reset_report_counts()
        if runtime_limit:
            stop_reason = "runtime limit"
            break
        if stagnation_limit:
            stop_reason = "quality plateau"
            break
    phase = "Done"
    report_progress(force=True)
    print(f"Finished: {stop_reason}.", flush=True)
    feasible_front = sorted(archive.front, key=lambda p: p.objectives)
    if not feasible_front:
        print("No matching itinerary found.")
    verification_started = monotonic()
    for ind in feasible_front:
        ind.rank = 0
        errors = check_solution(ind.phenotype)
        if errors or objective_values(ind.phenotype) != ind.objectives:
            raise RuntimeError(f"Final itinerary verification failed: {errors}")
    if metrics is not None:
        metrics.update(
            elapsed_seconds=monotonic() - started, generations=len(history) - 1,
            setup_seconds=setup_seconds, initialization_seconds=initialization_seconds,
            initialization_candidates=initialization_candidates, offspring_candidates=offspring_candidates,
            first_feasible_seconds=search.first_feasible_seconds,
            final_verification_seconds=monotonic() - verification_started,
            hypervolume=quality.hypervolume, compromise=quality.compromise,
            balanced_solution_scales=balance.scales, balanced_solution_weights=balance.weights,
            objective_minima=[min(p.objectives[i] for p in feasible_front) for i in range(3)] if feasible_front else None,
            generated_transport=search.generated, feasible_transport=search.feasible,
            distinct_genotypes=len(search.seen_genotypes), complete_transport=search.complete,
            distinct_transport=len(search.seen_transports),
            cache_hits=search.cache_hits, cache_misses=search.cache_misses,
            dominated_option_removals=search.pruned,
            option_build_seconds=search.option_seconds,
            candidate_scans=search.candidate_scans, decoder_invocations=search.invocations,
            complete_result_reuse=search.result_hits, result_cache_entries=len(search.result_cache),
            result_cache_evictions=search.result_evictions,
            cache_entries=len(search.cache), cache_evictions=search.cache_evictions,
            static_option_entries=len(search.option_table),
            static_option_count=sum(map(len, search.option_table.values())),
            duration_rejections=search.duration_rejections, daytime_rejections=search.daytime_rejections,
            runtime_cutoffs=search.runtime_hits,
            distinct_feasible_transports=len(search.feasible_transports),
            feasible_transports_per_second=len(search.feasible_transports) / max(verification_started - started, 1e-9),
            parent_group_count=len(parent_groups(pop)),
            largest_parent_group_share=max(map(len, parent_groups(pop))) / len(pop),
            history=history,
            final_mutation_rate=schedule.rate, final_success_ratio=schedule.success_ratio,
            observations=observations,
            archive_by_start=dict(sorted(Counter(p.start_day for p in feasible_front).items())))
    return feasible_front


def main(show_database=False):
    validate_config()
    initialize_database()
    if show_database:
        print_database_summary()
    last_start = date.fromisoformat(config.TRAVEL_RANGE[0]) + timedelta(days=departure_days()[-1])
    print(f"\nTrip planner | {config.TOTAL_DAYS} days | {config.NUM_DESTINATIONS} destinations")
    print(f"Departure: {config.TRAVEL_RANGE[0]} to {last_start}")
    balance = ObjectiveBalance.from_data()
    front = optimize(balance=balance)
    if not front:
        return
    print("\nFINAL RECOMMENDED ITINERARIES")
    for title, ind in select_best_options(front, balance).items():
        print_itinerary(title, ind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-objective trip planning with explicit calendar feasibility")
    parser.add_argument("--show-database", action="store_true")
    args = parser.parse_args()
    main(show_database=args.show_database)
