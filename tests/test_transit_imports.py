import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class TransitImportTests(unittest.TestCase):
    def run_python(self, *args, cwd=ROOT):
        result = subprocess.run(
            [sys.executable, *args], cwd=cwd, capture_output=True, text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_direct_script_from_another_directory(self):
        self.run_python(
            "-c",
            "import runpy, sys; "
            f"sys.path.insert(0, {str(SRC / 'api')!r}); "
            f"module = runpy.run_path({str(SRC / 'api' / 'train_api.py')!r}); "
            "assert module['CITIES']; assert module['EuropeTransitScraper']",
            cwd=ROOT / "tests",
        )

    def test_package_module(self):
        self.run_python(
            "-c",
            "import runpy; module = runpy.run_module('api.train_api'); "
            "assert module['CITIES']; assert module['EuropeTransitScraper']",
            cwd=SRC,
        )

    def test_import_with_main_and_data_loader(self):
        self.run_python(
            "-c",
            "import main; import data_loader; "
            "from api.train_api import ConnectionResult, EuropeTransitScraper; "
            "from api.train_helpers import db_provider, flix_provider, train_models; "
            "assert ConnectionResult is db_provider.ConnectionResult; "
            "assert ConnectionResult is flix_provider.ConnectionResult; "
            "assert train_models.CITIES; "
            "from unittest.mock import patch; "
            "client = EuropeTransitScraper(); "
            "db = patch.object(client.db_provider, 'fetch_connections', return_value=[]); "
            "flix = patch.object(client.flix_provider, 'fetch_connections', return_value=[]); "
            "db.start(); flix.start(); "
            "assert client.fetch_both('Stuttgart', 'Paris', '2026-12-10') == ([], [])",
            cwd=SRC,
        )


if __name__ == "__main__":
    unittest.main()
