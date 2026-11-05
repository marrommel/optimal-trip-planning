"""NSGA-II with city-linked nights/hotel choices and edge-linked transport genes."""

import random
from dataclasses import replace
from collections.abc import Callable
from typing import TYPE_CHECKING

import config
from accommodation import HotelFrontiers
from config import NUM_DESTINATIONS, DESTINATIONS
from data_loader import CITIES, HOTELS_DB
from decoder import MODES, TransportSearch, genotype_key, transport_key
from models import Chromosome, Phenotype
from objectives import evaluate, dominates
from timing import departure_days, minimum_nights

TOTAL_AVAILABLE_DESTINATIONS = len(DESTINATIONS)

if TYPE_CHECKING:
    from progress import MutationSchedule


def new_edge() -> tuple[float, int, float]:
    return random.choice((9.0, 15.0, 23.0)), random.randrange(len(MODES)), random.random()


def edge_genes(ind: Chromosome) -> dict:
    route = [0, *ind.route, 0]
    return dict(zip(zip(route, route[1:]), zip(ind.dep_time_pref, ind.mode_pref, ind.connection_choice)))


def set_edges(ind: Chromosome, inherited: dict):
    route = [0, *ind.route, 0]
    genes = [inherited[edge] if edge in inherited else new_edge() for edge in zip(route, route[1:])]
    ind.dep_time_pref, ind.mode_pref, ind.connection_choice = map(list, zip(*genes))


def repair_nights(ind: Chromosome):
    minima = [minimum_nights(CITIES[city]) for city in ind.route]
    ind.nights = [max(value, low) for value, low in zip(ind.nights, minima)]
    while sum(ind.nights) > config.TOTAL_DAYS - 1:
        donors = [i for i, low in enumerate(minima) if ind.nights[i] > low]
        if not donors:
            break
        ind.nights[random.choice(donors)] -= 1


def init_population(size: int, *, greedy: bool = True) -> list[Chromosome]:
    """Mix bounded greedy seeding with random plans; immigrants stay random."""
    from baselines import greedy_seeds  # Deferred: baselines builds genes with set_edges.
    required = [index for index, name in CITIES.items() if name in config.MUST_VISIT_DESTINATIONS]
    available_starts = list(departure_days())
    random.shuffle(available_starts)
    starts = [available_starts[i % len(available_starts)] for i in range(size)]
    random.shuffle(starts)
    population = greedy_seeds(int(size * config.GREEDY_SEED_FRACTION)) if greedy else []
    for ind, start_day in zip(population, starts):
        ind.start_day = start_day
    for start_day in starts[len(population):]:
        pool = [c for c in range(1, TOTAL_AVAILABLE_DESTINATIONS + 1) if c not in required]
        route = required + random.sample(pool, NUM_DESTINATIONS - len(required))
        random.shuffle(route)
        nights = [minimum_nights(CITIES[c]) for c in route]
        spare = max(0, config.TOTAL_DAYS - 1 - sum(nights))
        for _ in range(random.randint(0, spare)):
            nights[random.randrange(len(route))] += 1
        ind = Chromosome(route, nights, [], [], start_day=start_day)
        set_edges(ind, {})
        if config.OPTIMIZE_HOTELS:
            ind.hotel_choice = [random.random() for _ in route]
        population.append(ind)
    return population


def evaluate_candidate(ind: Chromosome, hotels: HotelFrontiers | None,
                       search: TransportSearch, reuse_transport: Phenotype | None = None) -> Chromosome:
    """Decode and score exactly one chromosome, including its hotel genes.

    `reuse_transport` supplies an already decoded transport phenotype for moves
    that cannot change transport: the result cache is keyed on the complete
    genotype, hotel genes included, so a hotel-only neighbour would otherwise
    pay for a full re-decode it cannot benefit from."""
    original = clone(ind)
    if reuse_transport is not None:
        if not (config.OPTIMIZE_HOTELS and hotels is not None):
            raise ValueError("Transport reuse applies only to hotel-gene moves")
        ind.phenotype = hotels.select(reuse_transport, ind.hotel_choice)
        evaluate(ind)
    else:
        transport = search.decode_and_encode(ind)
        ind.phenotype = (hotels.select(transport, ind.hotel_choice)
                         if config.OPTIMIZE_HOTELS and hotels is not None else transport)
        evaluate(ind)
        if ind.constraint_violation == 0:
            search.remember(original, ind, transport)
    search.record(original, ind)
    return ind


def evaluate_population(pop: list[Chromosome], hotels: HotelFrontiers | None = None,
                        search: TransportSearch | None = None,
                        on_evaluated: Callable[[list[Chromosome]], None] | None = None) -> None:
    """Evaluate exactly one itinerary per chromosome, including its hotel genes."""
    if config.OPTIMIZE_HOTELS and hotels is None:
        hotels = HotelFrontiers(HOTELS_DB)
    if search is None:
        search = TransportSearch()
    evaluated = []
    for ind in pop:
        if search.expired() and evaluated:
            break
        evaluate_candidate(ind, hotels, search)
        evaluated.append(ind)
        if on_evaluated is not None:
            on_evaluated([ind])
    pop[:] = evaluated


def initialize_population(size: int, hotels: HotelFrontiers | None,
                          search: TransportSearch,
                          on_evaluated: Callable[[list[Chromosome]], None] | None = None) -> list[Chromosome]:
    population = init_population(size)
    evaluate_population(population, hotels, search, on_evaluated)
    return population


def fast_non_dominated_sort(pop: list[Chromosome]) -> list[list[Chromosome]]:
    # Keep sorting bookkeeping out of chromosomes: cloning must not copy a population graph.
    counts = [0] * len(pop)
    dominated = [[] for _ in pop]
    for i, p in enumerate(pop):
        for j in range(i + 1, len(pop)):
            if dominates(p, pop[j]):
                dominated[i].append(j)
                counts[j] += 1
            elif dominates(pop[j], p):
                dominated[j].append(i)
                counts[i] += 1
    front = [i for i, count in enumerate(counts) if count == 0]
    fronts = []
    while front:
        rank = len(fronts)
        fronts.append([pop[i] for i in front])
        next_front = []
        for i in front:
            pop[i].rank = rank
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    next_front.append(j)
        front = next_front
    return fronts


def crowding_distance_assignment(front: list[Chromosome]):
    for ind in front:
        ind.crowding_distance = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return
    for axis in range(3):
        ordered = sorted(front, key=lambda x: x.objectives[axis])
        spread = ordered[-1].objectives[axis] - ordered[0].objectives[axis]
        if spread == 0:
            continue
        ordered[0].crowding_distance = ordered[-1].crowding_distance = float("inf")
        for i in range(1, len(ordered) - 1):
            ordered[i].crowding_distance += (
                ordered[i + 1].objectives[axis] - ordered[i - 1].objectives[axis]) / spread


def rank_population(pop: list[Chromosome]) -> list[list[Chromosome]]:
    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        crowding_distance_assignment(front)
    return fronts


def parent_groups(pop: list[Chromosome]) -> list[list[Chromosome]]:
    """Phenotypic niching. Many genotypes decode to the same transport itinerary
    (differing only in hotel variant). Selecting one representative per decoded
    niche prevents a single itinerary from crowding out structurally different
    plans — the ecological niche principle applied to the mating pool."""
    groups: dict[tuple, list[Chromosome]] = {}
    for ind in pop:
        p = ind.phenotype
        complete = (not ind.constraint_violation and p is not None
                    and len(p.connections) == len(ind.route) + 1 and len(p.visits) == len(ind.route))
        key = ("transport", transport_key(p)) if complete else ("genotype", genotype_key(ind))
        groups.setdefault(key, []).append(ind)
    return list(groups.values())


def selection_key(ind: Chromosome) -> tuple:
    return ind.constraint_violation, ind.rank, -ind.crowding_distance


def group_representative(group: list[Chromosome]) -> Chromosome:
    best = min(map(selection_key, group))
    return random.choice([ind for ind in group if selection_key(ind) == best])


def tournament_selection(pop: list[Chromosome], groups: list[list[Chromosome]] | None = None) -> Chromosome:
    if config.TRANSPORT_GROUP_SELECTION:
        groups = parent_groups(pop) if groups is None else groups
        if not groups:
            raise ValueError("Cannot select parents from an empty population")
        if len(groups) == 1:
            return group_representative(groups[0])
        p1, p2 = (group_representative(group) for group in random.sample(groups, 2))
    else:
        if len(pop) == 1:
            return pop[0]
        p1, p2 = random.sample(pop, 2)
    key1, key2 = (p1.rank, -p1.crowding_distance), (p2.rank, -p2.crowding_distance)
    if key1 == key2:
        return random.choice((p1, p2))
    return p1 if key1 < key2 else p2


def complementary_partner(first: Chromosome, pop: list[Chromosome], balance_scales: list[float],
                          groups: list[list[Chromosome]] | None = None, sample: int = 4) -> Chromosome:
    """"Gegensätze ziehen sich an": draw several ordinary tournament winners and
    keep the one whose normalised objective vector lies farthest from `first`,
    so recombination combines complementary strengths instead of near-duplicates."""
    candidates = [tournament_selection(pop, groups) for _ in range(max(1, sample))]
    feasible = [c for c in candidates if c.constraint_violation == 0]
    if feasible:
        candidates = feasible

    def distance(other: Chromosome) -> float:
        return sum(((a - b) / scale) ** 2 for a, b, scale
                   in zip(first.objectives, other.objectives, balance_scales) if scale)

    return max(candidates, key=distance)


def subset_order_crossover(p1_route, p2_route):
    n = len(p1_route)
    if n <= 1:
        return random.choice((p1_route, p2_route))[:]
    left, right = sorted(random.sample(range(n + 1), 2))
    child = [None] * n
    child[left:right] = p1_route[left:right]
    remaining = [c for c in p2_route if c not in child]
    for i in range(n):
        if child[i] is None:
            child[i] = remaining.pop(0)
    required = {i for i, name in CITIES.items() if name in config.MUST_VISIT_DESTINATIONS}
    for city in sorted(required - set(child)):
        replace = next(i for i, value in enumerate(child) if value not in required)
        child[replace] = city
    return child


def clone(ind: Chromosome) -> Chromosome:
    return Chromosome(ind.route[:], ind.nights[:], ind.dep_time_pref[:],
                      ind.mode_pref[:], ind.connection_choice[:], ind.start_day,
                      hotel_choice=ind.hotel_choice[:])


def crossover(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    if random.random() > config.CROSSOVER_RATE:
        return clone(p1), clone(p2)
    children = []
    for first, second in ((p1, p2), (p2, p1)):
        route = subset_order_crossover(first.route, second.route)
        child = Chromosome(route, [], [], [], start_day=random.choice((first.start_day, second.start_day)))
        for city in route:
            parent = random.choice([p for p in (first, second) if city in p.route])
            i = parent.route.index(city)
            child.nights.append(parent.nights[i])
            if config.OPTIMIZE_HOTELS:
                child.hotel_choice.append(parent.hotel_choice[i])
        edges1, edges2 = edge_genes(first), edge_genes(second)
        inherited = {edge: random.choice([mapping[edge] for mapping in (edges1, edges2) if edge in mapping])
                     for edge in edges1.keys() | edges2.keys()}
        set_edges(child, inherited)
        repair_nights(child)
        children.append(child)
    return children[0], children[1]


def mutate(ind: Chromosome, schedule: "MutationSchedule | None" = None):
    """Mutate in place. The schedule sets both how often genes change and how
    far a single change may move: above the baseline step plans are restructured,
    at the baseline the operators are unrestricted, and below it they only polish."""
    if schedule is None:
        from progress import MutationSchedule  # Deferred: progress imports nsga2.
        schedule = MutationSchedule()
    rate = schedule.rate
    inherited = edge_genes(ind)
    city_genes = dict(zip(ind.route, ind.nights))
    hotel_genes = dict(zip(ind.route, ind.hotel_choice))
    if random.random() < rate:
        ind.start_day = random.choice(departure_days())
    if len(ind.route) > 1 and random.random() < rate:
        i, j = sorted(random.sample(range(len(ind.route)), 2))
        operator = random.randrange(3)
        if operator and schedule.fine:
            j = min(j, i + 2)
        if operator == 0:
            ind.route[i], ind.route[j] = ind.route[j], ind.route[i]
        elif operator == 1:
            ind.route.insert(j, ind.route.pop(i))
        else:
            ind.route[i:j + 1] = reversed(ind.route[i:j + 1])
    if random.random() < rate:
        replaceable = [i for i, city in enumerate(ind.route) if CITIES[city] not in config.MUST_VISIT_DESTINATIONS]
        available = sorted(set(range(1, TOTAL_AVAILABLE_DESTINATIONS + 1)) - set(ind.route))
        if replaceable and available:
            city = random.choice(available)
            ind.route[random.choice(replaceable)] = city
            city_genes[city] = minimum_nights(CITIES[city])
            if config.OPTIMIZE_HOTELS:
                hotel_genes[city] = random.random()
    ind.nights = [city_genes[c] for c in ind.route]
    if config.OPTIMIZE_HOTELS:
        ind.hotel_choice = [random.random() if random.random() < rate else hotel_genes[c]
                            for c in ind.route]
    set_edges(ind, inherited)

    if len(ind.route) > 1 and random.random() < rate:
        donors = [i for i, city in enumerate(ind.route) if ind.nights[i] > minimum_nights(CITIES[city])]
        if donors:
            donor = random.choice(donors)
            receiver = random.choice([i for i in range(len(ind.route)) if i != donor])
            ind.nights[donor] -= 1
            ind.nights[receiver] += 1
    steps = (-2, -1, 1, 2) if schedule.coarse else (-1, 1)
    for i in range(len(ind.route)):
        if random.random() < rate:
            ind.nights[i] += random.choice(steps)
    repair_nights(ind)
    for i in range(len(ind.mode_pref)):
        if random.random() < rate:
            ind.dep_time_pref[i] = random.choice((9.0, 15.0, 23.0))
        if random.random() < rate:
            ind.mode_pref[i] = random.randrange(len(MODES))
        if random.random() < rate:
            ind.connection_choice[i] = random.random()
    if random.random() < config.COORDINATED_MUTATION_RATE:
        span = 3 if schedule.coarse else 2
        first = random.randrange(max(1, len(ind.mode_pref) + 1 - span))
        mode = random.randrange(len(MODES))
        window = random.choice((9.0, 15.0, 23.0))
        for edge in range(first, min(first + span, len(ind.mode_pref))):
            ind.mode_pref[edge] = mode
            ind.dep_time_pref[edge] = window
            ind.connection_choice[edge] = random.random()
        if random.random() < 0.5:
            ind.start_day = random.choice(departure_days())


def itinerary_key(ind: Chromosome) -> tuple:
    p = ind.phenotype
    if p is None:
        raise ValueError("Deduplication requires decoded individuals")
    if ind.constraint_violation:
        return (genotype_key(ind),
                tuple((h.city, h.property_token, h.hotel_name) for h in p.hotels))
    return decoded_itinerary_key(ind)


def decoded_itinerary_key(ind: Chromosome) -> tuple:
    p = ind.phenotype
    if p is None:
        raise ValueError("Decoded itinerary identity requires a phenotype")
    return (
        transport_key(p),
        tuple((h.city, h.hotel_name, h.property_token, h.check_in_day, h.nights) for h in p.hotels))


def unique_individuals(pop: list[Chromosome]) -> list[Chromosome]:
    unique = {}
    for ind in pop:
        unique.setdefault(itinerary_key(ind), ind)
    return list(unique.values())


def select_survivors(pop: list[Chromosome], size: int, hotels: HotelFrontiers | None = None,
                     on_evaluated: Callable[[list[Chromosome]], None] | None = None,
                     search: TransportSearch | None = None) -> list[Chromosome]:
    unique = unique_individuals(pop)
    count = len(parent_groups(unique)) if config.TRANSPORT_GROUP_SELECTION else len(unique)
    if count < size and (search is None or not search.expired()):
        immigrants = init_population(size - count, greedy=False)
        evaluate_population(immigrants, hotels, search, on_evaluated)
        unique = unique_individuals(unique + immigrants)
    if not unique:
        raise ValueError("Cannot select survivors without evaluated candidates")
    fronts = rank_population(unique)
    if config.TRANSPORT_GROUP_SELECTION:
        representatives = [group_representative(group) for group in parent_groups(unique)]
        random.shuffle(representatives)
        selected = sorted(representatives, key=selection_key)[:size]
        selected_ids = {id(ind) for ind in selected}
        extra = [ind for ind in unique if id(ind) not in selected_ids]
        random.shuffle(extra)
        selected.extend(sorted(extra, key=selection_key)[:size - len(selected)])
    else:
        selected = []
        for front in fronts:
            remaining = size - len(selected)
            if remaining <= 0:
                break
            selected.extend(sorted(front, key=lambda x: x.crowding_distance, reverse=True)[:remaining])
    # Tiny finite spaces may have fewer unique plans than the requested population.
    while len(selected) < size:
        # Preserve the evaluated hotel alternative; breeding always clones genes before mutation.
        selected.append(replace(random.choice(selected)))
    rank_population(selected)
    return selected
