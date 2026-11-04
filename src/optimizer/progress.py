"""Historical feasible archive and fixed-reference, exact three-objective quality metrics."""

from collections import deque
from dataclasses import dataclass
import math

import config
from models import Chromosome
from nsga2 import decoded_itinerary_key, itinerary_key
from objectives import ObjectiveBalance, objective_dominates, validate_balanced_weights

Point = tuple[float, float, float]


class ParetoArchive:
    def __init__(self):
        self.front: list[Chromosome] = []
        self.keys: set[tuple] = set()

    def update(self, candidates: list[Chromosome]):
        for ind in candidates:
            if ind.constraint_violation != 0:
                continue
            if any(not math.isfinite(v) or v < 0 for v in ind.objectives):
                raise ValueError("Archive objectives must be finite and non-negative")
            key = itinerary_key(ind)
            if key in self.keys or any(objective_dominates(p.objectives, ind.objectives) for p in self.front):
                continue
            retained = []
            for old in self.front:
                if objective_dominates(ind.objectives, old.objectives):
                    self.keys.remove(itinerary_key(old))
                else:
                    retained.append(old)
            self.front = [*retained, ind]
            self.keys.add(key)


def hypervolume(points: set[Point], reference: Point) -> float:
    """Exact union of boxes [point, reference]; inputs are already normalized."""
    if len(reference) != 3 or any(not math.isfinite(v) or v <= 0 for v in reference):
        raise ValueError("Hypervolume reference must be finite and positive")
    if any(len(p) != 3 or any(not math.isfinite(v) or v < 0 or v >= r for v, r in zip(p, reference))
           for p in points):
        raise ValueError("Fixed hypervolume reference must be strictly worse than every archive point; "
                         "increase HYPERVOLUME_REFERENCE consistently for all comparison runs")
    ordered = sorted(points)
    yz = []
    volume = 0.0
    for i, (x, y, z) in enumerate(ordered):
        yz.append((y, z))
        next_x = ordered[i + 1][0] if i + 1 < len(ordered) else reference[0]
        if next_x == x:
            continue
        lowest_z, area = reference[2], 0.0
        for py, pz in sorted(yz):
            if pz < lowest_z:
                area += (reference[1] - py) * (lowest_z - pz)
                lowest_z = pz
        volume += (next_x - x) * area
    return volume


class MutationSchedule:
    """Rechenberg's 1/5 success rule, adapted to a dominance-based success signal.

    While more than SUCCESS_TARGET of the offspring Pareto-dominate the parent
    they descend from, the search is still finding its way downhill, so the step
    grows; once such improvements become rare the step shrinks and the search
    refines what it already has. `coarse` and `fine` let the operators widen or
    narrow their structural changes, not only their probability.

    SUCCESS_TARGET must be calibrated against the dominance ratio this problem
    actually produces at the baseline rate: a target the search can never reach
    turns the controller into a one-way decay to MUTATION_RATE_MIN."""

    def __init__(self):
        self.rate = config.MUTATION_RATE
        self.scale = 1.0
        self.success_ratio: float | None = None

    def observe(self, successes: int, trials: int) -> None:
        """`trials` counts only offspring with a feasible parent to improve on;
        children of infeasible parents carry no information about step size."""
        if trials <= 0:
            self.success_ratio = None
            return
        self.success_ratio = successes / trials
        if self.success_ratio > config.SUCCESS_TARGET:
            self.scale *= config.STEP_GROWTH
        else:
            self.scale /= config.STEP_GROWTH
        self.rate = min(max(config.MUTATION_RATE * self.scale, config.MUTATION_RATE_MIN),
                        config.MUTATION_RATE_MAX)
        # Re-derive the scale from the clamped rate so a saturated bound cannot wind up.
        self.scale = self.rate / config.MUTATION_RATE if config.MUTATION_RATE else 1.0

    @property
    def coarse(self) -> bool:
        return self.scale > 1.0

    @property
    def fine(self) -> bool:
        """Only below the baseline step; at scale 1.0 the operators stay unrestricted."""
        return self.scale < 1.0


@dataclass(frozen=True)
class OffspringStats:
    generated: int
    decoded: int
    unique_feasible: int
    complete: int
    duplicate_complete: int

    @classmethod
    def measure(cls, candidates: list[Chromosome], generated: int):
        feasible = {decoded_itinerary_key(p) for p in candidates if p.constraint_violation == 0}
        complete = [p for p in candidates if p.phenotype is not None and not p.phenotype.errors
                    and len(p.phenotype.connections) == len(p.route) + 1
                    and (not config.OPTIMIZE_HOTELS or len(p.phenotype.hotels) == len(p.route))]
        duplicates = len(complete) - len({decoded_itinerary_key(p) for p in complete})
        return cls(generated, len(candidates), len(feasible), len(complete), duplicates)


@dataclass(frozen=True)
class QualitySnapshot:
    hypervolume: float
    compromise: float | None
    hv_gain: float | None
    compromise_gain: float | None
    archive_additions: int
    improved_regions: int
    window_generations: int
    stop: bool


class QualityMonitor:
    def __init__(self, balance: ObjectiveBalance | None = None):
        """Without a balance reference, only configuration validation is available."""
        self.scales = tuple(config.QUALITY_SCALES)
        self.raw_reference = tuple(config.HYPERVOLUME_REFERENCE)
        self.balance = balance
        self.window = config.QUALITY_WINDOW_GENERATIONS
        self.resolution = config.FRONT_PROGRESS_RESOLUTION
        self.hv_threshold = config.QUALITY_HV_MIN_GAIN
        self.compromise_threshold = config.QUALITY_COMPROMISE_MIN_GAIN
        if any(len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values)
               for values in (self.scales, self.raw_reference)):
            raise ValueError("Quality/reference scales must contain three finite positive values")
        validate_balanced_weights(config.BALANCED_SOLUTION_WEIGHTS if balance is None else balance.weights)
        if (type(self.window) is not int or self.window < 1
                or not math.isfinite(self.resolution) or self.resolution <= 0):
            raise ValueError("Quality window and objective-region resolution must be positive")
        if ((self.hv_threshold is None) != (self.compromise_threshold is None)
                or any(v is not None and (not math.isfinite(v) or v < 0)
                       for v in (self.hv_threshold, self.compromise_threshold))):
            raise ValueError("Set both calibrated quality thresholds to finite non-negative gains, or both to None")
        self.reference = tuple(v / s for v, s in zip(self.raw_reference, self.scales))
        self.history = deque(maxlen=self.window + 1)
        self.previous_keys: set[tuple] = set()
        self.previous_regions: set[tuple[int, ...]] = set()
        self.reset_report_counts()

    @property
    def stopping_enabled(self) -> bool:
        return self.hv_threshold is not None

    def reset_report_counts(self):
        self.additions = 0
        self.improved_regions = 0

    def observe(self, archive: ParetoArchive) -> QualitySnapshot:
        if self.balance is None:
            raise ValueError("Quality observation requires a balance reference derived before search")
        points = {tuple(v / s for v, s in zip(p.objectives, self.scales)) for p in archive.front}
        hv = hypervolume(points, self.reference)
        best = min((self.balance.score(p.objectives) for p in archive.front), default=None)
        regions = {tuple(math.floor(v / self.resolution) for v in p) for p in points}
        self.additions += len(archive.keys - self.previous_keys)
        # A new occupied cell is material only if no preceding cell is already as good in every axis.
        self.improved_regions += sum(
            not any(all(a <= b for a, b in zip(old, cell)) for old in self.previous_regions)
            for cell in regions - self.previous_regions)
        self.previous_regions = regions
        self.previous_keys = archive.keys.copy()
        self.history.append((hv, best))
        hv_gain = score_gain = None
        if len(self.history) == self.window + 1 and self.history[0][1] is not None and best is not None:
            hv_gain = max(0.0, hv - self.history[0][0])
            score_gain = max(0.0, self.history[0][1] - best)
        stop = (self.stopping_enabled and hv_gain is not None and score_gain is not None
                and hv_gain <= self.hv_threshold and score_gain <= self.compromise_threshold)
        return QualitySnapshot(hv, best, hv_gain, score_gain, self.additions,
                               self.improved_regions, self.window, stop)
