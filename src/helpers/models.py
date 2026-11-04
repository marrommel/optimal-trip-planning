from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import List, Tuple, Dict

@dataclass
class Connection:
    day: int           # Offset from the first date in TRAVEL_RANGE
    src: int
    dest: int
    mode: str          # 'train', 'bus', 'train+bus', 'flight'
    dep_hour: float
    duration: float    # hours
    cost: float        # € (Varies by day)
    co2: float         # kg
    
    # Metadata
    src_name: str = ""
    dest_name: str = ""
    provider: str = ""     
    station_src: str = ""
    station_dest: str = ""
    fare_source: str = ""
    quoted_at: str = ""
    arrival_time: datetime | None = None
    departure_time: datetime | None = None
    stops: int = 0
    journey_id: str = ""
    access_hours: float = 0.0
    egress_hours: float = 0.0
    co2_method: str = "unspecified estimate"

@dataclass
class Hotel:
    city: int
    tier: int          # 1=Hostel/1-2 stars, 2=Unclassified/3 stars, 3=4-5 stars
    daily_rates: Dict[int, float]  # Day -> one-night EUR total incl. supplied fees; missing = no quote.
    discomfort: float
    co2_per_night: float
    
    # Metadata
    city_name: str = ""
    hotel_name: str = ""
    synthetic: bool = False
    property_token: str = ""
    check_in_hour: float | None = None   # Earliest check-in (local hour); None = config default
    check_out_hour: float | None = None  # Latest check-out (local hour); None = config default

@dataclass
class HotelStay:
    """Represents the decoded, actual stay chosen by the optimizer."""
    hotel_name: str
    tier: int
    check_in_day: int
    nights: int
    total_cost: float
    total_co2: float
    discomfort_total: float
    city: int = -1
    synthetic: bool = False
    property_token: str = ""
    check_in_hour: float | None = None
    check_out_hour: float | None = None


@dataclass
class Visit:
    city: int
    arrival: datetime
    departure: datetime
    nights: int
    rest_nights: int
    usable_hours: float

@dataclass
class Chromosome:
    route: List[int]
    nights: List[int]
    dep_time_pref: List[float]
    mode_pref: List[int]
    connection_choice: List[float] = field(default_factory=list)
    start_day: int = 0  # Departure-date offset within TRAVEL_RANGE
    
    phenotype: 'Phenotype | None' = None
    objectives: Tuple[float, float, float] = (float('inf'), float('inf'), float('inf'))
    constraint_violation: float = 0.0
    rank: int = -1
    crowding_distance: float = 0.0
    violations: List[str] = field(default_factory=list)
    hotel_choice: List[float] = field(default_factory=list)  # City-linked keys in [0, 1].
    # Objectives of the parent this child descends from; None outside breeding or
    # when that parent was infeasible, which excludes the child from step control.
    parent_objectives: 'Tuple[float, float, float] | None' = None

@dataclass
class Phenotype:
    route_sequence: List[int]
    durations: List[int]
    connections: List[Connection]
    hotels: List[HotelStay]  # Empty if OPTIMIZE_HOTELS = False
    visits: List[Visit] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_day: int = 0
    search_status: str = "greedy"


def copy_phenotype(p: Phenotype) -> Phenotype:
    """Keep cached transport, visits and hotel alternatives independent of callers."""
    return replace(p, route_sequence=p.route_sequence[:], durations=p.durations[:],
                   connections=[replace(c) for c in p.connections],
                   hotels=[replace(h) for h in p.hotels], visits=[replace(v) for v in p.visits],
                   errors=p.errors[:])