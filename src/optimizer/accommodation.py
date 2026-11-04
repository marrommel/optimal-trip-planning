from dataclasses import replace
from datetime import date
from collections.abc import Callable, Iterable
import math
from typing import TypeVar

import config
from data_loader import CITIES
from models import Hotel, HotelStay, Phenotype, copy_phenotype
from objectives import objective_dominates, stay_clock
from timing import minimum_nights

T = TypeVar("T")


def stay_values(stay: HotelStay) -> tuple[float, float, float]:
    return stay.total_cost, stay.discomfort_total, stay.total_co2


def stay_key(stay: HotelStay) -> tuple[float, ...]:
    check_in, check_out = stay_clock(stay)
    return (*stay_values(stay), check_in, -check_out)


def nondominated(items: Iterable[T], values: Callable[[T], tuple[float, ...]]) -> list[T]:
    front: list[T] = []
    for item in items:
        point = values(item)
        if any(objective_dominates(values(other), point) for other in front):
            continue
        front = [other for other in front if not objective_dominates(point, values(other))]
        front.append(item)
    return front


class HotelFrontiers:
    """Precompute nondominated stays; each hotel gene selects one available stay.

    Per-(city, check-in day, nights) non-dominated frontier, precomputed once.
    Collapses a combinatorial accommodation sub-space into a small Pareto set
    before search begins, so the evolutionary budget is spent exclusively on
    decisions that are genuinely trade-offs.
    """

    def __init__(self, hotels: list[Hotel]):
        self.base_date, last = map(date.fromisoformat, config.TRAVEL_RANGE)
        self.options: dict[tuple[int, int, int], list[HotelStay]] = {}
        for city in sorted({h.city for h in hotels}):
            available = sorted(
                (h for h in hotels if h.city == city and (config.ALLOW_SYNTHETIC_HOTELS or not h.synthetic)),
                key=lambda h: (h.hotel_name, h.property_token))
            for day in range((last - self.base_date).days):
                for nights in range(minimum_nights(CITIES[city]),
                                    min(config.TOTAL_DAYS - 1, (last - self.base_date).days - day) + 1):
                    stays = [
                        HotelStay(h.hotel_name, h.tier, day, nights,
                                  sum(h.daily_rates[d] for d in range(day, day + nights)),
                                  h.co2_per_night * nights, h.discomfort * nights,
                                  city, h.synthetic, h.property_token, h.check_in_hour, h.check_out_hour)
                        for h in available if all(d in h.daily_rates for d in range(day, day + nights))
                    ]
                    if any(not math.isfinite(value) or value < 0 for stay in stays for value in stay_values(stay)):
                        raise ValueError("Hotel contributions must be finite and non-negative")
                    self.options[city, day, nights] = nondominated(stays, stay_key)

    def select(self, transport: Phenotype, choices: list[float]) -> Phenotype:
        if (len(choices) != len(transport.route_sequence) - 2
                or any(not 0 <= key <= 1 for key in choices)):
            raise ValueError("Hotel-choice genes must contain one key in [0, 1] per destination")
        result = copy_phenotype(transport)
        if result.errors:
            return result
        for visit, key in zip(result.visits, choices):
            interval = (visit.city, (visit.arrival.date() - self.base_date).days, visit.nights)
            options = self.options.get(interval, [])
            if not options:
                result.errors.append(f"No available hotel for the actual nights in {CITIES[visit.city]}")
                return result
            result.hotels.append(replace(options[min(int(key * len(options)), len(options) - 1)]))
        return result
