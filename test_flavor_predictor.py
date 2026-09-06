import unittest

from flavor_predictor import predict_flavor_experience


class FlavorPredictorTests(unittest.TestCase):
    def test_balances_taste_and_aroma(self) -> None:
        prediction = predict_flavor_experience(
            {
                "sucrose": 45,
                "citric_acid": 20,
                "limonene": 18,
                "linalool": 10,
            }
        )

        self.assertEqual(prediction.finish, "rounded sweet")
        self.assertGreater(prediction.primary_tastes["sweet"], prediction.primary_tastes["sour"])
        self.assertIn("citrus", prediction.aromatic_notes)
        self.assertIn("floral", prediction.aromatic_notes)
        self.assertAlmostEqual(prediction.overall_intensity, 0.233, places=3)

    def test_detects_warming_profile(self) -> None:
        prediction = predict_flavor_experience(
            {
                "caffeine": 40,
                "theobromine": 25,
                "capsaicin": 45,
                "pyrazines": 15,
            }
        )

        self.assertEqual(prediction.finish, "warming")
        self.assertIn("bitter", prediction.primary_tastes)
        self.assertIn("heat", prediction.sensations)
        self.assertIn("roasted", prediction.aromatic_notes)

    def test_clamps_out_of_range_inputs(self) -> None:
        prediction = predict_flavor_experience(
            {
                "sucrose": 200,
                "citric_acid": -10,
                "menthol": 150,
            }
        )

        self.assertEqual(prediction.primary_tastes["sweet"], 1.0)
        self.assertNotIn("sour", prediction.primary_tastes)
        self.assertEqual(prediction.sensations["cooling"], 1.0)
        self.assertEqual(prediction.overall_intensity, 1.0)


if __name__ == "__main__":
    unittest.main()
