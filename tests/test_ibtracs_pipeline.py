import unittest
import pandas as pd

from scripts.preprocess_ibtracs import haversine_km


class IbtracsPipelineTests(unittest.TestCase):
    def test_haversine_distance(self):
        self.assertAlmostEqual(float(haversine_km(0, 0, 0, 1)), 111.195, places=2)

    def test_storm_grouping_does_not_cross_boundary(self):
        df = pd.DataFrame({"SID": ["A", "A", "B", "B"], "LAT": [1, 2, 10, 11]})
        self.assertTrue(pd.isna(df.groupby("SID").LAT.shift().iloc[0]))
        self.assertTrue(pd.isna(df.groupby("SID").LAT.shift().iloc[2]))


if __name__ == "__main__":
    unittest.main()
