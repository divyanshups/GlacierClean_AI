"""
tests/test_loader.py
Tests for the data loader module.
"""

import io
import unittest
import pandas as pd
from core.data_loader import load_dataset, save_dataset


class TestDataLoader(unittest.TestCase):

    def _sample_csv(self) -> io.StringIO:
        csv_content = "name,age,salary\nAlice,30,50000\nBob,,60000\nAlice,30,50000\n"
        return io.StringIO(csv_content)

    def test_load_csv_from_stringio(self):
        df = load_dataset(self._sample_csv(), file_name="test.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 3))  # 3 data rows, header excluded

    def test_columns_preserved(self):
        df = load_dataset(self._sample_csv(), file_name="test.csv")
        self.assertListEqual(list(df.columns), ["name", "age", "salary"])

    def test_missing_preserved(self):
        df = load_dataset(self._sample_csv(), file_name="test.csv")
        self.assertEqual(df["age"].isnull().sum(), 1)

    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            load_dataset(io.StringIO("a,b\n1,2"), file_name="test.xyz")

    def test_save_and_reload(self):
        import tempfile, os
        df_orig = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_dataset(df_orig, path)
            df_loaded = load_dataset(path)
            pd.testing.assert_frame_equal(df_orig, df_loaded)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
