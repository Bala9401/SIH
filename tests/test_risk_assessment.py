import unittest

from prediction.risk_assessment import CycloneRiskAssessor


class RiskAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.assessor = CycloneRiskAssessor()
        self.satellite = {
            "available": True,
            "class_name": "insat3d_ir_cyclone_ds",
            "confidence": 0.9,
            "task": "satellite_product_classification",
            "intensity_prediction_available": False,
            "image_risk_score": 30.0,
        }

    def assess(self, wind, pressure, current, forecast):
        return self.assessor.assess_risk(wind, forecast, current, pressure, self.satellite)

    def test_risk_levels_change_with_conditions(self):
        self.assertEqual(self.assess(20, None, {"lat": 5, "lon": 50}, [] )["risk_level"], "LOW")
        self.assertEqual(self.assess(45, 995, {"lat": 14, "lon": 80}, [{"lat": 14, "lon": 80, "wind_estimated": 45}])["risk_level"], "MODERATE")
        strengthening = [{"lat": 20, "lon": 86, "wind_estimated": 80}, {"lat": 20, "lon": 86, "wind_estimated": 100}]
        self.assertEqual(self.assess(75, 970, {"lat": 20, "lon": 86}, strengthening)["risk_level"], "HIGH")
        very_high = [
            {"lat": 20, "lon": 86, "wind_estimated": 110, "uncertainty_radius_km": 300},
            {"lat": 20, "lon": 86, "wind_estimated": 150, "uncertainty_radius_km": 300},
        ]
        self.assertEqual(self.assess(110, 940, {"lat": 20, "lon": 86}, very_high)["risk_level"], "VERY HIGH")

    def test_missing_pressure_is_reported_and_weights_redistributed(self):
        result = self.assess(45, None, {"lat": 5, "lon": 50}, [{"lat": 5, "lon": 50, "wind_estimated": 45}])
        self.assertIsNone(result["factors"]["pressure_score"])
        self.assertNotIn("pressure_score", result["risk_weights"])
        self.assertIn("pressure unavailable", result["reason"])

    def test_satellite_product_is_context_only(self):
        result = self.assess(45, 995, {"lat": 14, "lon": 80}, [{"lat": 14, "lon": 80, "wind_estimated": 45}])
        self.assertEqual(result["risk_source"], "satellite_context_and_meteorological_track_analysis")
        self.assertEqual(result["satellite_contribution"], "bounded_contextual_score")
        self.assertEqual(result["satellite_analysis"]["task"], "satellite_product_classification")
        self.assertIn("satellite", result["risk_contribution"])
        self.assertEqual(result["image_risk_score"], 30.0)

    def test_image_analysis_changes_final_score(self):
        weak_image = dict(self.satellite, image_risk_score=7.5)
        strong_image = dict(self.satellite, image_risk_score=40.0)
        weak = CycloneRiskAssessor().assess_risk(
            45, [{"lat": 14, "lon": 80, "wind_estimated": 45}],
            {"lat": 14, "lon": 80}, 995, weak_image)
        strong = CycloneRiskAssessor().assess_risk(
            45, [{"lat": 14, "lon": 80, "wind_estimated": 45}],
            {"lat": 14, "lon": 80}, 995, strong_image)
        self.assertGreater(strong["risk_score"], weak["risk_score"])
        self.assertGreater(strong["risk_contribution"]["satellite"], weak["risk_contribution"]["satellite"])


if __name__ == "__main__":
    unittest.main()
