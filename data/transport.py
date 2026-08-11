from dataclasses import dataclass
from enum import Enum


class TransportType(Enum):
    PLANE = "flight"
    TRAIN = "train"
    BUS = "bus"

@dataclass
class Transport:
    id: int
    origin: str
    dest: str
    day: int
    price: float
    duration: int
    co2_kg: float
    type: TransportType