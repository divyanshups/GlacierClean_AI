import io
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load root conftest stubs explicitly for non-pytest execution.
importlib.import_module("conftest")


def _make_dirty_df(n: int = 200) -> pd.DataFrame:
    """Create a synthetic dirty DataFrame for testing."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": range(n),
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.normal(50_000, 15_000, n),
            "category": rng.choice(["A", "B", "C", None], n),
            "note": ["  hello  " if i % 10 == 0 else "world" for i in range(n)],
            "const": [42] * n,
        }
    )

    # Inject missing values.
    df.loc[rng.choice(n, 30, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "income"] = np.nan

    # Inject duplicates.
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    # Inject outliers.
    df.loc[df.index[:5], "income"] = 1_000_000
    return df


class TestPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run the pipeline once; share results across tests."""
        # Import here to avoid module-level failures if deps are missing.
        from core.pipeline import CleaningPipeline

        dirty = _make_dirty_df()
        buf = io.StringIO()
        dirty.to_csv(buf, index=False)
        buf.seek(0)

        pipeline = CleaningPipeline()
        cls.result = pipeline.run(buf, file_name="test.csv", save_output=False)

    # Basic structure.
    def test_result_keys(self):
        expected = {
            "df_raw",
            "df_cleaned",
            "report",
            "change_log",
            "issues",
            "actions",
            "profile",
        }
        self.assertEqual(set(self.result.keys()), expected)

    def test_cleaned_is_dataframe(self):
        self.assertIsInstance(self.result["df_cleaned"], pd.DataFrame)

    def test_cleaned_not_empty(self):
        self.assertGreater(len(self.result["df_cleaned"]), 0)

    # Quality improvements.
    def test_quality_score_improves(self):
        qs = self.result["report"]["quality_score"]
        self.assertGreaterEqual(qs["after"], qs["before"])

    def test_missing_rate_reduced(self):
        delta = self.result["report"]["delta"]["missing_rate"]
        self.assertLessEqual(delta["after"], delta["before"])

    def test_duplicates_removed(self):
        raw_dupes = self.result["df_raw"].duplicated().sum()
        clean_dupes = self.result["df_cleaned"].duplicated().sum()
        # After deduplication, cleaned dataset should have <= duplicates.
        self.assertLessEqual(clean_dupes, raw_dupes)

    def test_constant_column_dropped(self):
        self.assertNotIn("const", self.result["df_cleaned"].columns)

    # Report structure.
    def test_report_has_quality_score(self):
        self.assertIn("quality_score", self.result["report"])

    def test_issues_detected(self):
        self.assertGreater(len(self.result["issues"]), 0)

    def test_change_log_populated(self):
        self.assertGreater(len(self.result["change_log"]), 0)


if __name__ == "__main__":
    unittest.main()
