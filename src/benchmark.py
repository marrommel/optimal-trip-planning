"""Offline, equal-search-time genetic optimizer comparisons.

Run from the repository root:
    uv run python src\\benchmark.py --seconds 600 --seeds 11 42 97 --output comparison.json

Uses the configured trip and model without overriding budget or carbon limits.
Only the selected search variants, population, seed and stopping limits vary.
"""

import sys
from pathlib import Path
for _sub in ('', 'helpers', 'optimizer'):
    sys.path.insert(0, str(Path(__file__).resolve().parent / _sub))

import argparse
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
import io
import json
import math
import random
from statistics import median
from time import monotonic
from unittest.mock import patch

import config
import data_loader
import main
from decoder import transport_key

DEFAULT_VARIANTS = ["genetic"]


def variants() -> dict[str, dict]:
    current = {key: getattr(config, key) for key in (
        "COORDINATED_MUTATION_RATE", "TRANSPORT_GROUP_SELECTION", "COMPLEMENTARY_PAIRING",
        "PRUNE_EQUIVALENT_TRANSPORT", "TRANSPORT_CACHE_SIZE", "TRANSPORT_RESULT_CACHE_SIZE",
        "MUTATION_RATE_MIN", "MUTATION_RATE_MAX", "STEP_GROWTH", "SUCCESS_TARGET",
        "GREEDY_SEED_FRACTION", "LOCAL_SEARCH_FRACTION", "LOCAL_SEARCH_MAX_STEPS",
        "BOMB_STRENGTH", "BOMB_HV_MIN_GAIN")}
    # Every variant isolates one feature against a plain genetic baseline, so the
    # reference stays a plain GA no matter what settings.toml enables by default.
    plain = {**current, "GREEDY_SEED_FRACTION": 0.0, "LOCAL_SEARCH_FRACTION": 0.0,
             "BOMB_STRENGTH": 0}
    return dict(
        genetic=plain,
        coordinated={**plain, "COORDINATED_MUTATION_RATE": 0.2},
        complementary={**plain, "COMPLEMENTARY_PAIRING": True},
        transport_diversity={**plain, "TRANSPORT_GROUP_SELECTION": True},
        fixed_step={**plain, "MUTATION_RATE_MIN": config.MUTATION_RATE,
                    "MUTATION_RATE_MAX": config.MUTATION_RATE},
        no_pruning={**plain, "PRUNE_EQUIVALENT_TRANSPORT": False},
        no_cache={**plain, "TRANSPORT_CACHE_SIZE": 0, "TRANSPORT_RESULT_CACHE_SIZE": 0},
        no_result_reuse={**plain, "TRANSPORT_RESULT_CACHE_SIZE": 0},
        greedy_seed={**plain, "GREEDY_SEED_FRACTION": current["GREEDY_SEED_FRACTION"] or 0.2},
        local_search={**plain, "LOCAL_SEARCH_FRACTION": current["LOCAL_SEARCH_FRACTION"] or 0.1},
        ils={**plain, "GREEDY_SEED_FRACTION": current["GREEDY_SEED_FRACTION"] or 0.2,
             "LOCAL_SEARCH_FRACTION": current["LOCAL_SEARCH_FRACTION"] or 0.1,
             "BOMB_STRENGTH": current["BOMB_STRENGTH"] or 3},
    )


def load_offline():
    """Use normal cache validation, but fail instead of fetching or overwriting data."""
    with ExitStack() as stack:
        for owner, name in ((data_loader, "search_flights"), (data_loader, "search_hotels"),
                            (data_loader, "EuropeTransitScraper")):
            stack.enter_context(patch.object(owner, name, side_effect=RuntimeError(
                "Benchmark requires valid local caches; provider access is disabled")))
        data_loader.initialize_database()


def run_comparison(output: Path, seconds: float, seeds: list[int],
                   names: list[str], climate_target: float | None = None, population: int | None = None):
    if (not math.isfinite(seconds) or seconds <= 0
            or climate_target is not None and (not math.isfinite(climate_target) or climate_target < 0)):
        raise ValueError("Runtime must be positive and climate target non-negative, both finite")
    main.validate_config()
    population = main.POPULATION_SIZE if population is None else population
    if type(population) is not int or population < 2:
        raise ValueError("Benchmark population must be an integer >= 2")
    settings = variants()
    if any(name not in settings for name in names) or not seeds or not names:
        raise ValueError("Supply known variants and at least one seed")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {output}")
    loading_started = monotonic()
    load_offline()
    loading_seconds = monotonic() - loading_started
    normalization_started = monotonic()
    balance = main.ObjectiveBalance.from_data()
    normalization_seconds = monotonic() - normalization_started
    caches = ("flight_quotes.json", "transit_quotes.json", "hotel_quotes.csv", "hotel_quotes.json")
    document = {
        "search_seconds_per_run": seconds,
        "population_size": population,
        "generation_cap": 10**9,
        "quality_stopping": False,
        "climate_target": climate_target,
        "loading_seconds": loading_seconds,
        "normalization_seconds": normalization_seconds,
        "balanced_solution_scales": balance.scales,
        "balanced_solution_weights": balance.weights,
        "flight_emissions": "Google Flights itinerary grams / 1000; no extra aviation multiplier",
        "configuration": {key: value for key, value in vars(config).items() if key.isupper()},
        "cache_sha256": {name: sha256((data_loader.CACHE_DIR / name).read_bytes()).hexdigest()
                         for name in caches if (data_loader.CACHE_DIR / name).exists()},
        "source_sha256": {str(p.relative_to(Path(__file__).parent)): sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.rglob("*.py")},
        "method": "Genetic optimizer with unchanged trip/model; serial runs; loading and shared normalization "
                  "excluded, search setup included; "
                  "quality stopping disabled; independent final verification may extend the budget.",
        "runs": [],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
    for seed in seeds:
        order = names[:]
        random.Random(seed).shuffle(order)
        for name in order:
            metrics = {}
            with patch.multiple(config, **settings[name], RANDOM_SEED=seed,
                                MAX_RUNTIME_SECONDS=seconds, QUALITY_HV_MIN_GAIN=None,
                                QUALITY_COMPROMISE_MIN_GAIN=None), \
                    patch.object(main, "MAX_GENERATIONS", 10**9), \
                    patch.object(main, "POPULATION_SIZE", population), redirect_stdout(io.StringIO()):
                front = main.optimize(metrics, balance=balance)
            minimum = metrics["objective_minima"]
            result = dict(variant=name, seed=seed, settings=settings[name], **metrics,
                          reached_climate_target=(bool(minimum and minimum[2] <= climate_target)
                                                  if climate_target is not None else None),
                          archive_transport_count=len({transport_key(p.phenotype) for p in front}),
                          archive=[dict(start_day=p.start_day, route=p.route, nights=p.nights,
                                        objectives=p.objectives,
                                        modes=[c.mode for c in p.phenotype.connections],
                                        departures=[c.departure_time.isoformat() for c in p.phenotype.connections])
                                   for p in front])
            document["runs"].append(result)
            document["summary"] = summarize(document["runs"], names)
            with output.open("w", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"{name:12} seed={seed} HV={metrics['hypervolume']:.6f}"
                  f" minima={minimum} feasible={metrics['feasible_transport']}/{metrics['generated_transport']}"
                  f" seconds={metrics['elapsed_seconds']:.2f}", flush=True)


def summarize(runs: list[dict], names: list[str]) -> dict:
    """Median/range and paired differences; missing feasible results remain explicit."""
    fields = ("hypervolume", "compromise", "initialization_seconds", "first_feasible_seconds",
              "feasible_transports_per_second", "parent_group_count", "largest_parent_group_share",
              "candidate_scans", "complete_result_reuse", "runtime_cutoffs",
              "local_refinements", "local_bombs", "generations", "search_seconds",
              "local_search_candidates", "bomb_candidates",
              "final_mutation_rate", "final_success_ratio")
    summary = {}
    reference = {r["seed"]: r for r in runs if r["variant"] == names[0]}
    for name in names:
        selected = [r for r in runs if r["variant"] == name]
        values = {}
        for field in fields:
            present = [r[field] for r in selected if r.get(field) is not None]
            pairs = {r["seed"]: r[field] - reference[r["seed"]][field] for r in selected
                     if r["seed"] in reference and r.get(field) is not None
                     and reference[r["seed"]].get(field) is not None}
            values[field] = dict(median=median(present), minimum=min(present), maximum=max(present),
                                 observed=len(present), paired_differences=pairs) if present else None
        summary[name] = values
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 97])
    parser.add_argument("--variants", nargs="+", choices=list(variants()), default=DEFAULT_VARIANTS)
    parser.add_argument("--population", type=int, default=None, help="Override population for every comparison run")
    parser.add_argument("--climate-target", type=float, default=None, help="Optional policy-valid evaluation target")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_comparison(args.output, args.seconds, args.seeds, args.variants, args.climate_target, args.population)
