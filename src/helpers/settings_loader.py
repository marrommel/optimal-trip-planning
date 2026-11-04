"""Load optimizer settings and fixed reference data without silent overrides."""
import tomllib
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
INTERNAL_FILES = (
    _SRC / "optimizer" / "settings.toml",
    _SRC / "helpers" / "assumptions.toml",
)
# Quality-based stopping stays off unless explicitly configured.
DEFAULTS = {"QUALITY_HV_MIN_GAIN": None, "QUALITY_COMPROMISE_MIN_GAIN": None}


def load_internal_settings(namespace: dict) -> None:
    """Flatten section headers; reject duplicate names across sections/files."""
    settings = {}
    for path in INTERNAL_FILES:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        for key, value in data.items():
            # Lowercase tables are only section headers; uppercase ones are dict-valued settings.
            if not isinstance(value, dict):
                raise ValueError(f"Setting {key} in {path} must belong to a TOML section")
            entries = value if key.islower() else {key: value}
            for name, item in entries.items():
                if not name.isupper():
                    raise ValueError(f"Configuration key {name} in {path} must be uppercase")
                if name in settings or name in namespace:
                    raise ValueError(f"Duplicate configuration setting {name} in {path}")
                if name == "HOTEL_DISCOMFORT_PER_NIGHT_BY_STARS":
                    item = {int(star): weight for star, weight in item.items()}
                settings[name] = tuple(item) if isinstance(item, list) else item
    settings = DEFAULTS | settings
    conflicts = settings.keys() & namespace.keys()
    if conflicts:
        raise ValueError(f"Duplicate configuration settings: {', '.join(sorted(conflicts))}")
    namespace.update(settings)
