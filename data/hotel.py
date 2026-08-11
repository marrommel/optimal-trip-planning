from dataclasses import dataclass
from typing import Dict


@dataclass
class Hotel:
    name: str
    location: str
    prices: Dict[int, float]
    comfort: int