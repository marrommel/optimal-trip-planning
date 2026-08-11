import random
import copy
from typing import List, Dict, Any

from data.hotel import Hotel
from data.transport import Transport


class TripOptimizerGA:

    def __init__(self, transports: List[Transport], hotels: List[Hotel], destinations: List[str],
                 start_point: str, min_duration: int, max_duration: int, min_destinations: int,
                 pop_size: int = 100, generations: int = 100, mutation_rate: float = 0.2, elitism: int = 5):

        self.transports = transports
        self.hotels = hotels
        self.destinations = destinations
        self.start_point = start_point
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_destinations = min_destinations

        # GA Parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism

    def get_specific_transports(self, origin: str, dest: str, day: int) -> List[Transport]:
        return [t for t in self.transports if t.origin == origin and t.dest == dest and t.day == day]

    def get_hotels_at(self, location: str) -> List[Hotel]:
        return [h for h in self.hotels if h.location == location]

    def repair_genome(self, genome: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Walks through the chronological timeline of the chromosome.
        If crossover/mutation broke the physics of the trip (e.g., no transport on that day),
        this finds a new valid transport or alters the route to make it strictly valid.
        """
        current_loc = self.start_point
        current_day = 0

        for i, gene in enumerate(genome):
            if gene['type'] == 'stop':
                # 1. Fix Transport
                valid_trans = self.get_specific_transports(current_loc, gene['loc'], current_day)
                if not valid_trans:
                    # Fallback: Find ANY transport leaving current_loc on this day
                    any_trans = [t for t in self.transports if
                                 t.origin == current_loc and t.day == current_day and t.dest in self.destinations]
                    if any_trans:
                        chosen = random.choice(any_trans)
                        gene['loc'] = chosen.dest
                        gene['transport_in'] = chosen.id
                    else:
                        gene['transport_in'] = None  # Will be heavily penalized in fitness
                elif gene['transport_in'] not in [t.id for t in valid_trans]:
                    gene['transport_in'] = random.choice(valid_trans).id

                # 2. Fix Hotel
                valid_hotels = self.get_hotels_at(gene['loc'])
                if valid_hotels and gene['hotel'] not in [h.name for h in valid_hotels]:
                    gene['hotel'] = random.choice(valid_hotels).name

                current_loc = gene['loc']
                current_day += gene['duration']

            elif gene['type'] == 'return':
                valid_trans = self.get_specific_transports(current_loc, self.start_point, current_day)
                if valid_trans:
                    if gene['transport_in'] not in [t.id for t in valid_trans]:
                        gene['transport_in'] = random.choice(valid_trans).id
                else:
                    # If stranded, look for transport on a later day and extend the last stay
                    future_trans = [t for t in self.transports if
                                    t.origin == current_loc and t.dest == self.start_point and t.day >= current_day]
                    if future_trans:
                        best = min(future_trans, key=lambda x: x.day)
                        gene['transport_in'] = best.id
                        if i > 0:
                            genome[i - 1]['duration'] += (best.day - current_day)  # Extend previous stay
                    else:
                        gene['transport_in'] = None

        return genome

    def create_random_individual(self) -> List[Dict[str, Any]]:
        num_stops = random.randint(self.min_destinations, self.min_destinations + 2)
        genome = []
        for _ in range(num_stops):
            genome.append({
                'type': 'stop',
                'loc': random.choice(self.destinations),
                'hotel': '',
                'duration': random.randint(1, 3),
                'transport_in': ''
            })
        genome.append({'type': 'return', 'loc': self.start_point, 'transport_in': ''})
        return self.repair_genome(genome)

    def calculate_fitness(self, genome: List[Dict[str, Any]]) -> float:
        cost = 0.0
        co2 = 0.0
        comfort = 0.0
        penalty = 0.0

        current_day = 0
        destinations_visited = set()

        transport_dict = {t.id: t for t in self.transports}
        hotel_dict = {h.name: h for h in self.hotels}

        for gene in genome:
            if gene['transport_in'] is None:
                penalty += 10000  # Broken link penalty
            else:
                t = transport_dict[gene['transport_in']]
                cost += t.price
                co2 += t.co2_kg

            if gene['type'] == 'stop':
                destinations_visited.add(gene['loc'])
                if gene['hotel'] in hotel_dict:
                    h = hotel_dict[gene['hotel']]
                    comfort += h.comfort
                    # Sum hotel prices for the specific days we stay there
                    for d in range(current_day, current_day + gene['duration']):
                        cost += h.prices.get(d, h.prices[0])  # Fallback to day 0 price if missing
                else:
                    penalty += 10000
                current_day += gene['duration']

        # Constraints Penalities
        if current_day < self.min_duration or current_day > self.max_duration:
            penalty += abs(current_day - ((self.min_duration + self.max_duration) / 2)) * 1000

        if len(destinations_visited) < self.min_destinations:
            penalty += (self.min_destinations - len(destinations_visited)) * 2000

        # Multi-objective weights (Soft requirements)
        # $1 = 1 point. Let's say 1kg CO2 = $0.5 equivalent cost. 1 Comfort pt = $10 discount.
        fitness = cost + (co2 * 0.5) - (comfort * 10) + penalty
        return fitness

    def crossover(self, p1: List[Dict], p2: List[Dict]) -> List[Dict]:
        p1_stops = [g for g in p1 if g['type'] == 'stop']
        p2_stops = [g for g in p2 if g['type'] == 'stop']

        # Split and splice stops, maintaining start and return integrity
        if len(p1_stops) > 1 and len(p2_stops) > 1:
            cx1 = random.randint(1, len(p1_stops) - 1)
            cx2 = random.randint(1, len(p2_stops) - 1)
            child_stops = copy.deepcopy(p1_stops[:cx1]) + copy.deepcopy(p2_stops[cx2:])
        else:
            child_stops = copy.deepcopy(p1_stops) if random.random() < 0.5 else copy.deepcopy(p2_stops)

        child = child_stops + [{'type': 'return', 'loc': self.start_point, 'transport_in': None}]
        return self.repair_genome(child)  # Fix the broken dates/destinations!

    def mutate(self, genome: List[Dict]) -> List[Dict]:
        genome = copy.deepcopy(genome)
        for gene in genome:
            if gene['type'] == 'stop' and random.random() < self.mutation_rate:
                mutation_type = random.choice(['duration', 'location', 'hotel'])
                if mutation_type == 'duration':
                    gene['duration'] = max(1, gene['duration'] + random.choice([-1, 1]))
                elif mutation_type == 'location':
                    gene['loc'] = random.choice([d for d in self.destinations if d != gene['loc']])
                elif mutation_type == 'hotel':
                    # Erase it, let repair_genome pick a new valid one
                    gene['hotel'] = ''
        return self.repair_genome(genome)

    def run(self):
        population = [self.create_random_individual() for _ in range(self.pop_size)]

        for gen in range(self.generations):
            # Evaluate fitness (lower is better)
            pop_fitness = [(ind, self.calculate_fitness(ind)) for ind in population]
            pop_fitness.sort(key=lambda x: x[1])

            next_generation = [x[0] for x in pop_fitness[:self.elitism]]  # Elitism

            while len(next_generation) < self.pop_size:
                # Tournament selection
                t1 = random.choice(pop_fitness[:int(self.pop_size / 2)])
                t2 = random.choice(pop_fitness[:int(self.pop_size / 2)])
                parent1, parent2 = t1[0], t2[0]

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation

            if gen % 10 == 0 or gen == self.generations - 1:
                best_fit = pop_fitness[0][1]
                print(f"Generation {gen}: Best Fitness Score = {best_fit:.2f}")

        # Return best solution found
        final_fitness = [(ind, self.calculate_fitness(ind)) for ind in population]
        final_fitness.sort(key=lambda x: x[1])
        return final_fitness[0][0]
