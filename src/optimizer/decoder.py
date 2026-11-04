"""Decode genetic choices into one itinerary using cached timetable queries."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import math
from time import monotonic
from typing import cast

import config
from data_loader import CITIES, CONNECTIONS_BY_ROUTE
from models import Chromosome, Connection, Phenotype, Visit, copy_phenotype
from objectives import connection_objectives, objective_dominates
from timing import city_zone, departure_days, journey_eligible, make_visit, minimum_nights, travel_window, trip_bounds, utc

MODES = ("train", "bus", "flight", "train+bus")


def time_window(hour: float) -> int:
    return 0 if 6 <= hour < 12 else 1 if 12 <= hour < 18 else 2


def decode(chromosome: Chromosome, search: "TransportSearch | None" = None) -> Phenotype:
    route = [0, *chromosome.route, 0]
    result = Phenotype(route, chromosome.nights[:], [], [], start_day=chromosome.start_day)
    n = len(chromosome.route)
    if (len(set(chromosome.route)) != n or any(c not in CITIES or c == 0 for c in chromosome.route)
            or n == 0):
        result.errors.append("Route must contain distinct known destinations")
        return result
    if (len(chromosome.nights) != n or len(chromosome.mode_pref) != n + 1
            or len(chromosome.dep_time_pref) != n + 1
            or len(chromosome.connection_choice) != n + 1):
        raise ValueError("Chromosome dimensions do not match its route")
    if any(not isinstance(value, int) or value < minimum_nights(CITIES[city])
           for city, value in zip(chromosome.route, chromosome.nights)):
        result.errors.append("Allocated nights are below the city minimum or are not integers")
        return result
    if any(not 0 <= key <= 1 for key in chromosome.connection_choice):
        raise ValueError("Connection-choice keys must be in [0, 1]")
    if any(mode not in range(len(MODES)) for mode in chromosome.mode_pref):
        raise ValueError("Unknown transport mode gene")
    if type(chromosome.start_day) is not int or chromosome.start_day not in departure_days():
        result.errors.append("Departure date is outside the allowed trip-start window")
        return result
    return (search if search is not None else TransportSearch()).greedy(chromosome, result)


def connection_key(conn: Connection) -> tuple:
    return (conn.src, conn.dest, conn.mode, conn.departure_time, conn.arrival_time,
            conn.station_src, conn.station_dest, conn.cost, conn.co2, conn.provider,
            conn.journey_id, conn.stops, conn.access_hours, conn.egress_hours)


def transport_key(p: Phenotype) -> tuple:
    return tuple(connection_key(c) for c in p.connections)


def genotype_key(ind: Chromosome) -> tuple:
    return (tuple(ind.route), tuple(ind.nights), ind.start_day, tuple(ind.mode_pref),
            tuple(ind.dep_time_pref), tuple(ind.connection_choice), tuple(ind.hotel_choice))


@dataclass(frozen=True)
class TimedOption:
    connection: Connection
    leave: datetime
    arrive: datetime
    values: tuple[float, float, float]
    arrival_day: int


class TransportSearch:
    """Run-local timetable caches; genetic choices are followed without backtracking."""

    def __init__(self, stop_at: float | None = None):
        self.stop_at = stop_at
        self.cache_size = config.TRANSPORT_CACHE_SIZE
        self.prune = config.PRUNE_EQUIVALENT_TRANSPORT
        self.cache: OrderedDict[tuple, object] = OrderedDict()
        self.result_cache: OrderedDict[tuple, tuple[Phenotype, tuple]] = OrderedDict()
        self.result_cache_size = config.TRANSPORT_RESULT_CACHE_SIZE
        self.option_table: dict[tuple[int, int, int], tuple[TimedOption, ...]] = {}
        self.calendar: dict[tuple[str, date], tuple[datetime, ...]] = {}
        self.base_date = date.fromisoformat(config.TRAVEL_RANGE[0])
        self.zones = {city: city_zone(name) for city, name in CITIES.items()}
        self.bounds = {day: trip_bounds(day) for day in departure_days()}
        self.by_day: dict[tuple[int, int, int], tuple[Connection, ...]] = {}
        self.duration_rejections = 0
        for key, connections in CONNECTIONS_BY_ROUTE.items():
            if any(not math.isfinite(v) or v < 0 for c in connections for v in (c.cost, c.co2)):
                raise ValueError("Transport data require finite, non-negative fares and emissions")
            eligible = tuple(replace(c) for c in connections if journey_eligible(c))
            self.duration_rejections += len(connections) - len(eligible)
            self.by_day[key] = eligible
        self.cache_hits = self.cache_misses = self.pruned = 0
        self.cache_evictions = self.result_hits = self.result_evictions = self.invocations = 0
        self.candidate_scans = self.daytime_rejections = self.runtime_hits = 0
        self.generated = self.feasible = self.complete = 0
        self.seen_transports: set[tuple] = set()
        self.seen_genotypes: set[tuple] = set()
        self.option_seconds = 0.0
        self.started = monotonic()
        self.first_feasible_seconds: float | None = None
        self.feasible_transports: set[tuple] = set()

    def expired(self) -> bool:
        return self.stop_at is not None and monotonic() >= self.stop_at

    def memo[T](self, key: tuple, build: Callable[[], T]) -> T:
        if self.cache_size and key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            # Each namespace is used with one builder/value type.
            return cast(T, self.cache[key])
        self.cache_misses += 1
        value = build()
        if self.cache_size:
            self.cache[key] = value
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
                self.cache_evictions += 1
        return value

    def options(self, day: int, src: int, dest: int) -> tuple[TimedOption, ...]:
        key = (day, src, dest)
        if key in self.option_table:
            self.cache_hits += 1
            return self.option_table[key]
        started = monotonic()
        groups: dict[tuple, list[TimedOption]] = {}
        for conn in self.by_day.get(key, ()):
            leave, arrive = map(utc, travel_window(conn))
            values = connection_objectives(conn)
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Transport data require finite, non-negative connection contributions")
            option = TimedOption(conn, leave, arrive, values,
                                 (arrive.astimezone(self.zones[dest]).date() - self.base_date).days)
            # Preserve mode/window gene semantics and distinct equal-value offers.
            equivalent = (leave, arrive, conn.mode, conn.dep_hour,
                          conn.station_src, conn.station_dest, conn.stops,
                          conn.departure_time, conn.arrival_time, conn.access_hours, conn.egress_hours)
            group = groups.setdefault(equivalent, [])
            if self.prune and any(objective_dominates(old.values, values) for old in group):
                self.pruned += 1
                continue
            if self.prune:
                retained = [old for old in group if not objective_dominates(values, old.values)]
                self.pruned += len(group) - len(retained)
                group[:] = retained
            group.append(option)
        options = [option for group in groups.values() for option in group]
        options.sort(key=lambda o: (
            utc(o.connection.departure_time), utc(o.connection.arrival_time), o.connection.cost,
            o.connection.co2, o.connection.station_src, o.connection.station_dest,
            o.connection.provider, o.connection.journey_id))
        self.option_seconds += monotonic() - started
        self.cache_misses += 1
        result = tuple(options)
        # Only actual timetable keys enter the finite static table.
        if self.cache_size and key in self.by_day:
            self.option_table[key] = result
        return result

    def feasible_options(self, ind: Chromosome, i: int, day: int, previous: datetime | None
                         ) -> tuple[tuple[TimedOption, Visit | None], ...]:
        route = [0, *ind.route, 0]
        src, dest = route[i:i + 2]
        nights = ind.nights[i - 1] if i else 0
        def build():
            start, deadline = self.bounds[ind.start_day]
            start_utc, deadline_utc = utc(start), utc(deadline)
            result: list[tuple[TimedOption, Visit | None]] = []
            visits_by_leave: dict[datetime, Visit] = {}
            for option in self.options(day, src, dest):
                leave, arrive = option.leave, option.arrive
                if leave < start_utc or arrive >= deadline_utc:
                    continue
                if i == 0 and leave.astimezone(city_zone(config.START_CITY)).date() != start.date():
                    continue
                if dest == 0 and arrive.astimezone(city_zone(config.START_CITY)).date() != deadline.date() - timedelta(days=1):
                    continue
                visit = None
                if previous is not None:
                    if leave not in visits_by_leave:
                        visits_by_leave[leave] = make_visit(src, CITIES[src], previous, leave, self.calendar)
                    visit = visits_by_leave[leave]
                    if visit.usable_hours < config.MIN_USABLE_HOURS_PER_CITY:
                        self.daytime_rejections += 1
                    if (leave <= previous or visit.nights != nights or visit.rest_nights < nights
                            or visit.usable_hours < config.MIN_USABLE_HOURS_PER_CITY):
                        continue
                result.append((option, visit))
            return tuple(result)
        return self.memo(("feasible", ind.start_day, src, dest, nights, day, previous), build)

    def ordered(self, ind: Chromosome, i: int, candidates: tuple[tuple[TimedOption, Visit | None], ...]
                ) -> list[tuple[TimedOption, Visit | None]]:
        mode = MODES[ind.mode_pref[i]]
        window = time_window(ind.dep_time_pref[i])
        groups: list[list[tuple[TimedOption, Visit | None]]] = [[], [], [], []]
        for item in candidates:
            conn = item[0].connection
            group = (0 if conn.mode == mode else 2) + (time_window(conn.dep_hour) != window)
            groups[group].append(item)
        ordered = []
        for group in groups:
            if group:
                index = min(int(ind.connection_choice[i] * len(group)), len(group) - 1)
                ordered.extend(group[index:] + group[:index])
        return ordered

    def greedy(self, ind: Chromosome, result: Phenotype) -> Phenotype:
        day = ind.start_day
        previous = None
        for i, dest in enumerate(result.route_sequence[1:]):
            if self.expired():
                self.runtime_hits += 1
                return replace(result, errors=["Transport runtime limit reached; feasibility is unknown"],
                               search_status="runtime_limit")
            candidates = self.feasible_options(ind, i, day, previous)
            self.candidate_scans += len(candidates)
            if not candidates:
                src = result.route_sequence[i]
                result.errors.append(
                    f"No time-feasible connection on day {day}: {CITIES[src]} -> {CITIES[dest]}")
                return result
            option, visit = self.ordered(ind, i, candidates)[0]
            result.connections.append(replace(option.connection))
            if visit is not None:
                result.visits.append(replace(visit))
            previous = option.arrive
            if dest != 0:
                day = option.arrival_day + ind.nights[i]
        result.search_status = "feasible"
        return result

    def decode_and_encode(self, ind: Chromosome) -> Phenotype:
        if self.expired():
            self.runtime_hits += 1
            return Phenotype([0, *ind.route, 0], ind.nights[:], [], [],
                             errors=["Transport runtime limit reached; feasibility is unknown"],
                             start_day=ind.start_day, search_status="runtime_limit")
        key = genotype_key(ind)
        if self.result_cache_size and key in self.result_cache:
            self.result_hits += 1
            self.result_cache.move_to_end(key)
            p, genes = self.result_cache[key]
            ind.mode_pref, ind.dep_time_pref, ind.connection_choice = map(list, genes)
            return copy_phenotype(p)
        self.invocations += 1
        result = decode(ind, self)
        if result.search_status == "feasible" and not self.expired():
            self.encode(ind, result)
        return result

    def remember(self, original: Chromosome, encoded: Chromosome, p: Phenotype) -> None:
        if not self.result_cache_size or self.expired() or p.search_status != "feasible":
            return
        key = genotype_key(original)
        genes = (tuple(encoded.mode_pref), tuple(encoded.dep_time_pref), tuple(encoded.connection_choice))
        self.result_cache[key] = (copy_phenotype(p), genes)
        self.result_cache.move_to_end(key)
        if len(self.result_cache) > self.result_cache_size:
            self.result_cache.popitem(last=False)
            self.result_evictions += 1

    def encode(self, ind: Chromosome, p: Phenotype) -> None:
        """Write actual selections back into inherited edge genes after preference fallbacks.

        Lamarckian inheritance. The decoder resolves gene preferences into the
        actual timetable entry chosen; encode writes that realized choice back
        into the genotype, so an acquired, phenotypically validated trait becomes
        heritable. Nature's second optimizer — learning — layered onto evolution.
        """
        previous = None
        for i, conn in enumerate(p.connections):
            candidates = self.feasible_options(ind, i, conn.day, previous)
            group = [o.connection for o, _ in candidates
                     if o.connection.mode == conn.mode and time_window(o.connection.dep_hour) == time_window(conn.dep_hour)]
            ind.mode_pref[i] = MODES.index(conn.mode)
            ind.dep_time_pref[i] = conn.dep_hour
            ind.connection_choice[i] = (group.index(conn) + 0.5) / len(group)
            previous = utc(travel_window(conn)[1])

    def record(self, original: Chromosome, candidate: Chromosome) -> None:
        self.generated += 1
        self.seen_genotypes.add(genotype_key(original))
        feasible = candidate.constraint_violation == 0
        self.feasible += feasible
        p = candidate.phenotype
        if p is None or len(p.connections) != len(original.route) + 1 or len(p.visits) != len(original.route):
            return
        key = transport_key(p)
        self.complete += 1
        self.seen_transports.add(key)
        if feasible:
            self.feasible_transports.add(key)
            if self.first_feasible_seconds is None:
                self.first_feasible_seconds = monotonic() - self.started
