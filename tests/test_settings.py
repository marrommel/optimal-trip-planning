import io
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from test_optimizer import config, main
from helpers import settings_loader


class SettingsTests(unittest.TestCase):
    def test_optimizer_has_one_settings_file_and_no_legacy_keys(self):
        optimizer_dir = Path(__file__).resolve().parents[1] / "src" / "optimizer"
        self.assertEqual(sorted(p.name for p in optimizer_dir.glob("*.toml")), ["settings.toml"])
        loaded = {}
        settings_loader.load_internal_settings(loaded)
        for key in ("EVALUATION_MODE", "DECODER_MAX_NODES", "DIRECTED_SEED_FRACTION",
                    "DIRECTED_SEED_ATTEMPTS", "RETURN_LEG_REFINEMENT", "ROUTE_AWARE_SEEDING",
                    "MAX_VARIANTS_PER_INDIVIDUAL", "HOTEL_COMBINATION_CACHE_SIZE",
                    "RECOMMENDATION_WEIGHTS", "RECOMMENDATION_SCALES"):
            self.assertNotIn(key, loaded)
            self.assertFalse(hasattr(config, key))
        self.assertIsNone(loaded["QUALITY_HV_MIN_GAIN"])
        self.assertIsNone(loaded["QUALITY_COMPROMISE_MIN_GAIN"])
        self.assertIsInstance(loaded["BALANCED_SOLUTION_WEIGHTS"], tuple)
        sections = tomllib.loads((optimizer_dir / "settings.toml").read_text(encoding="utf-8"))
        self.assertNotIn("recommendation", sections)
        self.assertEqual(set(sections["objective_balancing"]), {"BALANCED_SOLUTION_WEIGHTS"})
        self.assertEqual(set(loaded["HOTEL_DISCOMFORT_PER_NIGHT_BY_STARS"]), set(range(6)))
        self.assertIn("London", loaded["CITY_TIMEZONES"])
        main.validate_config()

    def test_settings_file_values_reach_config_without_overrides(self):
        for path in settings_loader.INTERNAL_FILES:
            sections = tomllib.loads(path.read_text(encoding="utf-8"))
            for section, values in sections.items():
                entries = values if section.islower() else {section: values}
                for key, expected in entries.items():
                    if isinstance(expected, list):
                        expected = tuple(expected)
                    if key == "HOTEL_DISCOMFORT_PER_NIGHT_BY_STARS":
                        expected = {int(k): v for k, v in expected.items()}
                    self.assertEqual(getattr(config, key), expected, key)

    def test_duplicates_across_sections_files_and_trip_inputs_fail_atomically(self):
        cases = [
            (["[search]\nRANDOM_SEED=1\n[other]\nRANDOM_SEED=2"], {}),
            (["[search]\nRANDOM_SEED=1", "[reference]\nRANDOM_SEED=2"], {}),
            (["[search]\nRANDOM_SEED=1"], {"RANDOM_SEED": 3}),
            (["[search]\nRANDOM_SEED=1"], {"QUALITY_HV_MIN_GAIN": 0.1}),
        ]
        for contents, namespace in cases:
            with self.subTest(contents=contents, namespace=namespace), tempfile.TemporaryDirectory() as directory:
                paths = []
                for index, text in enumerate(contents):
                    path = Path(directory) / f"{index}.toml"
                    path.write_text(text, encoding="utf-8")
                    paths.append(path)
                before = namespace.copy()
                with patch.object(settings_loader, "INTERNAL_FILES", paths):
                    with self.assertRaisesRegex(ValueError, "Duplicate configuration"):
                        settings_loader.load_internal_settings(namespace)
                self.assertEqual(namespace, before)

    def test_explicit_quality_thresholds_override_only_defaults(self):
        content = b"[quality_monitoring]\nQUALITY_HV_MIN_GAIN=0.01\nQUALITY_COMPROMISE_MIN_GAIN=0.02\n"
        with patch.object(settings_loader, "INTERNAL_FILES", [Path("settings.toml")]), \
                patch.object(Path, "open", return_value=io.BytesIO(content)):
            namespace = {}
            settings_loader.load_internal_settings(namespace)
        self.assertEqual(namespace, {"QUALITY_HV_MIN_GAIN": 0.01, "QUALITY_COMPROMISE_MIN_GAIN": 0.02})

    def test_invalid_section_structure_is_reported(self):
        for content in (b"RANDOM_SEED=42", b"[search]\nrandom_seed=42"):
            with self.subTest(content=content), \
                    patch.object(settings_loader, "INTERNAL_FILES", [Path("settings.toml")]), \
                    patch.object(Path, "open", return_value=io.BytesIO(content)):
                with self.assertRaises(ValueError):
                    settings_loader.load_internal_settings({})

    def test_invalid_search_controls_fail_before_search(self):
        for owner, key, values in (
                (main, "POPULATION_SIZE", (True, 1, 2.5)),
                (main, "MAX_GENERATIONS", (True, 0, 2.5)),
                (config, "MAX_RUNTIME_SECONDS", (0, -1, float("inf"), float("nan"))),
                (config, "RANDOM_SEED", (True, 1.5))):
            for value in values:
                with self.subTest(key=key, value=value), patch.object(owner, key, value):
                    with self.assertRaisesRegex(ValueError, key):
                        main.validate_config()


if __name__ == "__main__":
    unittest.main()
