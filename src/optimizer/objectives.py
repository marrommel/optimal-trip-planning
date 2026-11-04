"""Pareto objectives and independent, plain-Python feasibility verification."""

from datetime import date, timedelta
from dataclasses import dataclass
import math
from statistics import median

import config
from data_loader import CITIES, CONNECTIONS_BY_ROUTE, HOTELS_DB
from models import Chromosome, Connection, Phenotype
from timing import ClockWindows, clock_window_overlap, city_zone, departure_days, elapsed_hours, journey_eligible, local_time, make_visit, minimum_nights, sleep_overlap, travel_window, trip_bounds, utc


def objective_dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def validate_balanced_weights(weights: tuple[float, ...]) -> None:
    if (len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights)
            or not math.isfinite(sum(weights)) or sum(weights) <= 0):
        raise ValueError("BALANCED_SOLUTION_WEIGHTS must contain three finite non-negative weights with positive sum")


@dataclass(frozen=True)
class ObjectiveBalance:
    """A trip-specific reference frozen before search, independent of the evolving front."""

    scales: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scales", tuple(self.scales))
        object.__setattr__(self, "weights", tuple(self.weights))
        if len(self.scales) != 3 or any(not math.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("Derived balance scales must contain three finite positive values")
        validate_balanced_weights(self.weights)

    @classmethod
    def from_data(cls) -> "ObjectiveBalance":
        """Estimate typical trip totals using equally weighted route and city medians.

        Transport: median of per-route objective medians, times destination count + 1.
        Hotels: median of per-city property medians, times TOTAL_DAYS - 1.
        These are reference magnitudes, not bounds or a constructed itinerary.
        """
        routes: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        calendar: ClockWindows = {}
        for connections in CONNECTIONS_BY_ROUTE.values():
            for conn in connections:
                if journey_eligible(conn):
                    values = connection_objectives(conn, calendar)
                    if any(not math.isfinite(v) or v < 0 for v in values):
                        raise ValueError("Cannot derive balance scales from invalid transport objectives")
                    routes.setdefault((conn.src, conn.dest), []).append(values)
        if not routes:
            raise ValueError("Cannot derive balance scales: no duration-eligible transport quotes are loaded")

        def typical(groups: list[list[tuple[float, float, float]]]) -> tuple[float, ...]:
            medians = tuple(median(median(point[axis] for point in group) for group in groups)
                            for axis in range(3))
            # Preserve unit invariance when a majority of quotes have a zero contribution.
            return tuple(value if value > 0 else max(point[axis] for group in groups for point in group)
                         for axis, value in enumerate(medians))

        transport = typical(list(routes.values()))
        cities: dict[int, list[tuple[float, float, float]]] = {}
        if config.OPTIMIZE_HOTELS:
            for hotel in HOTELS_DB:
                if hotel.city == 0 or not hotel.daily_rates or (hotel.synthetic and not config.ALLOW_SYNTHETIC_HOTELS):
                    continue
                values = (median(hotel.daily_rates.values()), hotel.discomfort, hotel.co2_per_night)
                if any(not math.isfinite(v) or v < 0 for v in (*hotel.daily_rates.values(), *values)):
                    raise ValueError("Cannot derive balance scales from invalid hotel objectives")
                cities.setdefault(hotel.city, []).append(values)
        hotel = typical(list(cities.values())) if cities else (0.0, 0.0, 0.0)
        totals = tuple(t * (config.NUM_DESTINATIONS + 1) + h * (config.TOTAL_DAYS - 1)
                       for t, h in zip(transport, hotel))
        # An identically zero reference contribution needs a neutral divisor, not division by zero.
        return cls(tuple(value if value > 0 else 1.0 for value in totals),
                   tuple(config.BALANCED_SOLUTION_WEIGHTS))

    def score(self, values: tuple[float, ...]) -> float:
        if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Balanced scoring requires three finite non-negative objectives")
        return sum(w * (value / scale) ** 2 for w, value, scale in zip(self.weights, values, self.scales))


def climate_impact(conn: Connection) -> float:
    """Use the cached API emissions, already scaled to the travel party at load time."""
    return conn.co2


def connection_objectives(conn: Connection, calendar: ClockWindows | None = None) -> tuple[float, float, float]:
    leave, arrive = travel_window(conn)
    hours = elapsed_hours(leave, arrive)
    discomfort = hours * config.TRAVEL_DISCOMFORT_PER_HOUR[conn.mode]
    discomfort += config.TRANSFER_DISCOMFORT * conn.stops
    discomfort += config.SLEEP_DISRUPTION_PER_HOUR * sleep_overlap(leave, arrive, CITIES[conn.src], calendar)
    origin = CITIES[conn.src]
    discomfort += config.DAYTIME_TRAVEL_DISCOMFORT_PER_HOUR[conn.mode] * clock_window_overlap(
        leave, arrive, origin, config.DAYTIME_START_HOUR, config.DAYTIME_END_HOUR, calendar)
    discomfort += config.NIGHT_TRAVEL_DISCOMFORT_PER_HOUR[conn.mode] * clock_window_overlap(
        leave, arrive, origin, config.NIGHT_START_HOUR, config.NIGHT_END_HOUR, calendar)
    discomfort += config.LONG_JOURNEY_DISCOMFORT_PER_HOUR * max(
        0, hours - config.LONG_JOURNEY_THRESHOLD_HOURS)
    if conn.mode == "flight":
        discomfort += config.FLIGHT_DISCOMFORT
    return conn.cost, discomfort, climate_impact(conn)


def stay_clock(stay) -> tuple[float, float]:
    """Earliest check-in and latest check-out hour; provider values override config defaults."""
    return (config.CHECK_IN_HOUR if stay is None or stay.check_in_hour is None else stay.check_in_hour,
            config.CHECK_OUT_HOUR if stay is None or stay.check_out_hour is None else stay.check_out_hour)


def hotel_wait_discomfort(visit, stay) -> float:
    """Early arrival waits per hour beyond a grace period; late departure costs one fixed penalty."""
    check_in_hour, check_out_hour = stay_clock(stay)
    check_in = local_time(visit.arrival.date(), check_in_hour, CITIES[visit.city])
    check_out = local_time(visit.departure.date(), check_out_hour, CITIES[visit.city])
    early = max(0.0, elapsed_hours(visit.arrival, check_in) - config.CHECK_IN_FREE_HOURS)
    late = elapsed_hours(check_out, visit.departure) > config.CHECK_OUT_FREE_HOURS
    return config.CHECK_IN_WAIT_DISCOMFORT_PER_HOUR * early + config.CHECK_OUT_WAIT_DISCOMFORT * late


def objective_values(p: Phenotype) -> tuple[float, float, float]:
    values = [connection_objectives(conn) for conn in p.connections]
    cost, discomfort, carbon = (sum(v[axis] for v in values) for axis in range(3))
    if config.OPTIMIZE_HOTELS:
        for stay in p.hotels:
            cost += stay.total_cost
            carbon += stay.total_co2
            discomfort += stay.discomfort_total
        stays = {stay.city: stay for stay in p.hotels}
        for visit in p.visits:
            discomfort += hotel_wait_discomfort(visit, stays.get(visit.city))
    return cost, discomfort, carbon


def check_solution(p: Phenotype) -> list[str]:
    """Recompute hard rules from connections, not from decoder feasibility flags."""
    errors = list(p.errors)
    route = p.route_sequence
    if not route or route[0] != 0 or route[-1] != 0:
        errors.append("Trip must start and finish at home")
    cities = route[1:-1]
    if (len(cities) != config.NUM_DESTINATIONS or len(cities) != len(set(cities))
            or any(c not in CITIES or c == 0 for c in cities)):
        errors.append("Wrong number of destinations, repeated city, or unknown city")
        return errors
    if not set(config.MUST_VISIT_DESTINATIONS).issubset({CITIES[c] for c in cities}):
        errors.append("A required destination was omitted")
    if len(p.connections) != len(route) - 1:
        errors.append("Incomplete transport itinerary")
        return errors
    if type(p.start_day) is not int or p.start_day not in departure_days():
        errors.append("Departure date is outside the allowed trip-start window")
        return errors
    if p.connections[0].day != p.start_day:
        errors.append("First departure must be on the selected start date")
    start, deadline = trip_bounds(p.start_day)
    for i, conn in enumerate(p.connections):
        if (conn.src, conn.dest) != (route[i], route[i + 1]):
            errors.append(f"Disconnected route at leg {i + 1}")
        if conn not in CONNECTIONS_BY_ROUTE.get((conn.day, conn.src, conn.dest), []):
            errors.append(f"Leg {i + 1} is not in the supplied timetable")
        if conn.departure_time is None or conn.arrival_time is None:
            errors.append(f"Missing timestamps on leg {i + 1}")
            continue
        leave, arrive = travel_window(conn)
        if utc(leave) < utc(start) or utc(arrive) >= utc(deadline):
            errors.append(f"Leg {i + 1} exceeds the inclusive calendar-day travel window")
        if elapsed_hours(conn.departure_time, conn.arrival_time) <= 0:
            errors.append(f"Non-positive journey duration on leg {i + 1}")
        if elapsed_hours(conn.departure_time, conn.arrival_time) > config.MAX_TRANSFER_HOURS:
            errors.append(f"Journey duration exceeds {config.MAX_TRANSFER_HOURS:g} hours on leg {i + 1}")
    if any(c.departure_time is None or c.arrival_time is None for c in p.connections):
        return errors
    leave_home = travel_window(p.connections[0])[0].astimezone(city_zone(config.START_CITY))
    return_home = travel_window(p.connections[-1])[1].astimezone(city_zone(config.START_CITY))
    if leave_home.date() != start.date():
        errors.append("Home departure must be on the selected first calendar day")
    if return_home.date() != deadline.date() - timedelta(days=1):
        errors.append("Home return must be on the exact last calendar day")
    if len(p.visits) != len(cities):
        errors.append("Missing destination stay records")
    if config.OPTIMIZE_HOTELS and len(p.hotels) != len(cities):
        errors.append("Missing accommodation")
    base_date = date.fromisoformat(config.TRAVEL_RANGE[0])
    for i, city in enumerate(cities):
        arrival = travel_window(p.connections[i])[1]
        departure = travel_window(p.connections[i + 1])[0]
        actual = make_visit(city, CITIES[city], arrival, departure)
        if utc(departure) <= utc(arrival):
            errors.append(f"Departure before arrival in {CITIES[city]}")
        if actual.rest_nights < minimum_nights(CITIES[city]):
            errors.append(f"Insufficient rested destination nights in {CITIES[city]}")
        if actual.usable_hours < config.MIN_USABLE_HOURS_PER_CITY:
            errors.append(f"Insufficient usable daytime in {CITIES[city]}")
        if i >= len(p.durations) or actual.nights != p.durations[i]:
            errors.append(f"Allocated nights do not match actual dates in {CITIES[city]}")
        if i < len(p.visits) and p.visits[i] != actual:
            errors.append(f"Incorrect stay summary in {CITIES[city]}")
        if config.OPTIMIZE_HOTELS and i < len(p.hotels):
            stay = p.hotels[i]
            check_in_day = (actual.arrival.date() - base_date).days
            days = range(check_in_day, check_in_day + actual.nights)
            hotel = next((h for h in HOTELS_DB if h.city == city and h.hotel_name == stay.hotel_name
                          and h.property_token == stay.property_token), None)
            if (hotel is None or not all(d in hotel.daily_rates for d in days)
                    or (hotel.synthetic and not config.ALLOW_SYNTHETIC_HOTELS)):
                errors.append(f"Hotel unavailable or unverified in {CITIES[city]}")
            elif (stay.city != city or stay.check_in_day != check_in_day or stay.nights != actual.nights
                  or not math.isclose(stay.total_cost, sum(hotel.daily_rates[d] for d in days))
                  or not math.isclose(stay.total_co2, hotel.co2_per_night * actual.nights)
                  or not math.isclose(stay.discomfort_total, hotel.discomfort * actual.nights)):
                errors.append(f"Incorrect hotel dates or totals in {CITIES[city]}")
    cost, _, carbon = objective_values(p)
    if cost > config.MAX_BUDGET:
        errors.append("Budget exceeded")
    if carbon > config.MAX_CO2:
        errors.append("Climate-impact allowance exceeded")
    return errors


def evaluate(ind: Chromosome):
    """MAX_BUDGET and MAX_CO2 are deliberately soft, relative penalties, so that
    infeasible intermediates can act as bridges between separated feasible
    regions. Structural errors are counted as hard integer penalties."""
    if ind.phenotype is None:
        raise ValueError("Decode the chromosome before evaluation")
    ind.objectives = objective_values(ind.phenotype)
    ind.violations = check_solution(ind.phenotype)
    cost, _, carbon = ind.objectives
    structural = [v for v in ind.violations if v not in ("Budget exceeded", "Climate-impact allowance exceeded")]
    ind.constraint_violation = (
        len(structural)
        + max(0, cost - config.MAX_BUDGET) / config.MAX_BUDGET
        + max(0, carbon - config.MAX_CO2) / config.MAX_CO2)


def dominates(p: Chromosome, q: Chromosome) -> bool:
    if p.constraint_violation or q.constraint_violation:
        return p.constraint_violation < q.constraint_violation
    return objective_dominates(p.objectives, q.objectives)
