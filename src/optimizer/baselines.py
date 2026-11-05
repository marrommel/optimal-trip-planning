"""Greedy construction of initial plans from per-route timetable medians.

Construction only: bounded quote sampling, no decoding or timetable search.
Sample medians guide route construction; they do not guarantee feasibility.
"""

import random
import logging
from bisect import bisect_right
from statistics import median

import config
from data_loader import CITIES, CONNECTIONS_BY_ROUTE
from models import Chromosome, Connection
from objectives import connection_objectives
from timing import departure_days, journey_eligible, minimum_nights

# Cost, discomfort, CO2 - the axes a single-objective greedy plan can specialise in.
AXES = (0, 1, 2, None)
MAX_QUOTES_PER_PAIR = 64


def route_weights() -> dict[tuple[int, int], tuple[float, float, float]]:
    """Estimate route medians without rescoring the entire timetable."""
    groups: dict[tuple[int, int], list[list[Connection]]] = {}
    for (_, src, dest), connections in CONNECTIONS_BY_ROUTE.items():
        if connections:
            groups.setdefault((src, dest), []).append(connections)
    weights = {}
    calendar = {}
    for pair, batches in groups.items():
        ends = []
        total = 0
        for batch in batches:
            total += len(batch)
            ends.append(total)
        points = []
        count = min(total, MAX_QUOTES_PER_PAIR)
        for i in range(count):
            offset = i * total // count
            batch_index = bisect_right(ends, offset)
            previous = ends[batch_index - 1] if batch_index else 0
            conn = batches[batch_index][offset - previous]
            if journey_eligible(conn):
                points.append(connection_objectives(conn, calendar))
        if points:
            weights[pair] = (median(p[0] for p in points), median(p[1] for p in points),
                             median(p[2] for p in points))
    return weights


def axis_scales(weights: dict[tuple[int, int], tuple[float, float, float]]) -> tuple[float, ...]:
    """Per-axis magnitudes so the weighted variant compares euros with kilograms."""
    if not weights:
        return (1.0, 1.0, 1.0)
    return tuple(median(point[axis] for point in weights.values()) or 1.0 for axis in range(3))


def edge_cost(weights, scales, axis: int | None, src: int, dest: int) -> float:
    point = weights.get((src, dest))
    if point is None:
        return float("inf")
    if axis is not None:
        return point[axis]
    return sum(w * value / scale for w, value, scale
               in zip(config.BALANCED_SOLUTION_WEIGHTS, point, scales))


def greedy_route(weights, scales, axis: int | None) -> list[int]:
    """Nearest neighbour from home, keeping enough slots for every must-visit city."""
    # Read the destination counts through nsga2 so greedy and random individuals
    # are always drawn from exactly the same city pool.
    import nsga2
    required = {index for index, name in CITIES.items() if name in config.MUST_VISIT_DESTINATIONS}
    unvisited = set(range(1, nsga2.TOTAL_AVAILABLE_DESTINATIONS + 1))
    route: list[int] = []
    current = 0
    while len(route) < nsga2.NUM_DESTINATIONS:
        remaining_slots = nsga2.NUM_DESTINATIONS - len(route)
        pending = required & unvisited
        pool = sorted(pending if len(pending) >= remaining_slots else unvisited)
        costs = [edge_cost(weights, scales, axis, current, city) for city in pool]
        best = min(costs)
        # Unquoted pairs leave the choice undetermined; fall back to an unbiased draw.
        reachable = [city for city, cost in zip(pool, costs) if cost == best]
        current = random.choice(reachable)
        route.append(current)
        unvisited.discard(current)
    return route


def greedy_plan(weights, scales, axis: int | None) -> Chromosome:
    """A greedy route with randomised nights, departure date and transport genes."""
    from nsga2 import set_edges  # Deferred: nsga2 seeds the population from here.
    route = greedy_route(weights, scales, axis)
    nights = [minimum_nights(CITIES[city]) for city in route]
    spare = max(0, config.TOTAL_DAYS - 1 - sum(nights))
    for _ in range(random.randint(0, spare)):
        nights[random.randrange(len(route))] += 1
    ind = Chromosome(route, nights, [], [], start_day=random.choice(departure_days()))
    set_edges(ind, {})
    if config.OPTIMIZE_HOTELS:
        ind.hotel_choice = [random.random() for _ in route]
    return ind


def greedy_seeds(count: int) -> list[Chromosome]:
    """Cycle the specialised axes so the seeds spread across the front, not into one corner."""
    if count <= 0:
        return []
    weights = route_weights()
    if not weights:
        logging.getLogger(__name__).warning(
            "No eligible sampled quotes for greedy seeding; using random initialization")
        return []
    scales = axis_scales(weights)
    return [greedy_plan(weights, scales, AXES[i % len(AXES)]) for i in range(count)]
