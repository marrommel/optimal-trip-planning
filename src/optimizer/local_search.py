"""First-improvement local search over a decoded plan's neighbourhood.

Lamarckian, like `TransportSearch.encode`: `improve` returns the refined
chromosome itself, so what the hill climber learns is inherited by the
offspring bred from it rather than being discarded after evaluation.

Hotel-only moves reuse transport; other moves use the memoized decoder.
The deadline is checked before every evaluation.
"""

from collections.abc import Callable, Iterator
from dataclasses import replace

import config
from accommodation import HotelFrontiers
from data_loader import CITIES
from decoder import MODES, TransportSearch, genotype_key, time_window
from models import Chromosome, Phenotype, copy_phenotype
from nsga2 import clone, edge_genes, evaluate_candidate, repair_nights, set_edges
from objectives import ObjectiveBalance, dominates
from timing import departure_days, minimum_nights

DEPARTURE_WINDOWS = (9.0, 15.0, 23.0)

Move = tuple[Chromosome, Phenotype | None]


def transport_only(ind: Chromosome) -> Phenotype | None:
    """The decoded itinerary without its accommodation, or None if unusable."""
    p = ind.phenotype
    if (p is None or p.errors or p.search_status != "feasible"
            or len(p.visits) != len(ind.route) or len(p.connections) != len(ind.route) + 1):
        return None
    return replace(copy_phenotype(p), hotels=[])


def hotel_moves(ind: Chromosome, hotels: HotelFrontiers | None) -> Iterator[Move]:
    """Step each hotel gene to the neighbouring entry of its own frontier.

    Transport cannot change, so the decoded itinerary is handed on for reuse."""
    transport = transport_only(ind) if config.OPTIMIZE_HOTELS and hotels is not None else None
    if (hotels is None or transport is None or ind.phenotype is None
            or len(ind.phenotype.hotels) != len(ind.route)):
        return
    for position, stay in enumerate(ind.phenotype.hotels):
        options = hotels.options.get((stay.city, stay.check_in_day, stay.nights), [])
        current = next((i for i, option in enumerate(options)
                        if (option.hotel_name, option.property_token)
                        == (stay.hotel_name, stay.property_token)), None)
        if current is None:
            continue
        for index in (current - 1, current + 1):
            if 0 <= index < len(options):
                candidate = clone(ind)
                candidate.hotel_choice[position] = (index + 0.5) / len(options)
                yield candidate, transport


def mode_moves(ind: Chromosome) -> Iterator[Move]:
    for edge in range(len(ind.mode_pref)):
        for mode in range(len(MODES)):
            if mode != ind.mode_pref[edge]:
                candidate = clone(ind)
                candidate.mode_pref[edge] = mode
                yield candidate, None


def window_moves(ind: Chromosome) -> Iterator[Move]:
    for edge in range(len(ind.dep_time_pref)):
        for hour in DEPARTURE_WINDOWS:
            if time_window(hour) != time_window(ind.dep_time_pref[edge]):
                candidate = clone(ind)
                candidate.dep_time_pref[edge] = hour
                yield candidate, None


def night_moves(ind: Chromosome) -> Iterator[Move]:
    """Move one night from every donor above its minimum to every other city."""
    minima = [minimum_nights(CITIES[city]) for city in ind.route]
    for donor in range(len(ind.route)):
        if ind.nights[donor] <= minima[donor]:
            continue
        for receiver in range(len(ind.route)):
            if receiver == donor:
                continue
            candidate = clone(ind)
            candidate.nights[donor] -= 1
            candidate.nights[receiver] += 1
            repair_nights(candidate)
            yield candidate, None


def start_day_moves(ind: Chromosome) -> Iterator[Move]:
    for start_day in (ind.start_day - 1, ind.start_day + 1):
        if start_day in departure_days():
            candidate = clone(ind)
            candidate.start_day = start_day
            yield candidate, None


def two_opt_moves(ind: Chromosome) -> Iterator[Move]:
    """Reverse each segment, including the single swap for a two-city route."""
    if len(ind.route) < 2:
        return
    inherited = edge_genes(ind)
    city_nights = dict(zip(ind.route, ind.nights))
    city_hotels = dict(zip(ind.route, ind.hotel_choice))
    for i in range(len(ind.route) - 1):
        for j in range(i + 1, len(ind.route)):
            candidate = clone(ind)
            candidate.route[i:j + 1] = reversed(candidate.route[i:j + 1])
            # Nights and hotel keys belong to the city, transport genes to the edge.
            candidate.nights = [city_nights[city] for city in candidate.route]
            if config.OPTIMIZE_HOTELS:
                candidate.hotel_choice = [city_hotels[city] for city in candidate.route]
            route = [0, *candidate.route, 0]
            # New directed edges use the old positional preference deterministically.
            # Random edge creation would change the neighbourhood on each sweep.
            fallback = list(zip(ind.dep_time_pref, ind.mode_pref, ind.connection_choice))
            edges = {edge: inherited.get(edge, fallback[k])
                     for k, edge in enumerate(zip(route, route[1:]))}
            set_edges(candidate, edges)
            yield candidate, None


def neighbours(ind: Chromosome, hotels: HotelFrontiers | None = None) -> Iterator[Move]:
    """Cloned, modified chromosomes ordered by evaluation cost, cheapest first.

    The second element is a decoded transport itinerary the caller may reuse, or
    None when the move forces a fresh decode."""
    origin = genotype_key(ind)
    generators = (hotel_moves(ind, hotels), mode_moves(ind), window_moves(ind),
                  night_moves(ind), start_day_moves(ind), two_opt_moves(ind))
    for generator in generators:
        for candidate, transport in generator:
            if genotype_key(candidate) != origin:
                yield candidate, transport


def better(candidate: Chromosome, incumbent: Chromosome, balance: ObjectiveBalance) -> bool:
    """Constrained dominance decides; the scalar compromise only breaks ties.

    The scalar fallback can still favor the compromise region. The archive
    retains nondominated alternatives even when the local walk rejects them."""
    p = candidate.phenotype
    if p is None or p.search_status == "runtime_limit":
        return False
    if dominates(candidate, incumbent):
        return True
    if dominates(incumbent, candidate):
        return False
    return balance.score(candidate.objectives) < balance.score(incumbent.objectives)


def improve(ind: Chromosome, search: TransportSearch, hotels: HotelFrontiers | None,
            balance: ObjectiveBalance, max_steps: int | None = None,
            on_evaluated: Callable[[list[Chromosome]], None] | None = None) -> Chromosome:
    """First-improvement hill climbing within the remaining runtime budget."""
    steps = config.LOCAL_SEARCH_MAX_STEPS if max_steps is None else max_steps
    best = ind
    for _ in range(steps):
        if search.expired():
            break
        accepted = None
        for candidate, transport in neighbours(best, hotels):
            if search.expired():
                break
            evaluate_candidate(candidate, hotels, search, transport)
            if on_evaluated is not None:
                on_evaluated([candidate])
            if better(candidate, best, balance):
                accepted = candidate
                break
        if accepted is None:
            break
        # Step control compares a child with the parent it descends from, which
        # a refined child still does; cloning drops the reference.
        accepted.parent_objectives = best.parent_objectives
        best = accepted
    return best


def refine(offspring: list[Chromosome], search: TransportSearch, hotels: HotelFrontiers | None,
           balance: ObjectiveBalance, fraction: float | None = None,
           on_evaluated: Callable[[list[Chromosome]], None] | None = None) -> int:
    """Refine the most promising share of a batch in place; returns how many."""
    share = config.LOCAL_SEARCH_FRACTION if fraction is None else fraction
    count = round(len(offspring) * share)
    if count <= 0 or search.expired():
        return 0
    order = sorted(range(len(offspring)),
                   key=lambda i: (offspring[i].constraint_violation,
                                  balance.score(offspring[i].objectives)))
    refined = 0
    for i in order[:count]:
        if search.expired():
            break
        offspring[i] = improve(offspring[i], search, hotels, balance, on_evaluated=on_evaluated)
        refined += 1
    return refined
