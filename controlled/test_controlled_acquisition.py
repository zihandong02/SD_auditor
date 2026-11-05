"""Fast deterministic checks; from the repository root run python -m unittest discover -s controlled."""
import unittest
from pathlib import Path

import numpy as np

import controlled_acquisition as experiment


class ControlledAcquisitionTests(unittest.TestCase):
    def test_exact_moments_and_budgets(self):
        for g, h in experiment.SCENARIOS.values():
            mu = np.sqrt(h)
            low, high = mu / 2, mu + 2 * g / mu
            probability = h / (h + 4 * g)
            np.testing.assert_allclose(low * (1 - probability) + high * probability, mu)
            np.testing.assert_allclose(
                (low - mu)**2 * (1 - probability) + (high - mu)**2 * probability, g
            )
            for gold in (False, True):
                a, s = experiment.oracle(g, h, gold=gold)
                self.assertAlmostEqual(float(np.mean(2 * a + s)), 0.6, places=12)
                self.assertTrue(np.all((a >= 0.01) & (a <= s) & (s <= 1)))

    def test_grid_size_and_feasibility(self):
        a, s = experiment.candidate_rules()
        self.assertEqual(len(a), 8447)
        self.assertTrue(np.all(np.mean(2 * a + s, axis=1) <= 0.6 + 1e-12))

    def test_unobserved_labels_cannot_affect_inference(self):
        rng = np.random.default_rng(12)
        g, h = experiment.SCENARIOS["local"]
        x, v, y = experiment.generate(rng, 900, g, h)
        a, s = experiment.oracle(g, h)
        uniform = rng.random(len(x))
        complete, compared = uniform < a[x], uniform < s[x]
        y_masked, v_masked = y.copy(), v.copy()
        y_masked[~complete] = np.nan
        v_masked[~compared] = -10000
        fit = experiment.fit_nuisance(x, v, y, complete, a)
        masked_fit = experiment.fit_nuisance(x, v_masked, y_masked, complete, a)
        for expected, actual in zip(fit, masked_fit):
            np.testing.assert_allclose(expected, actual)
        np.testing.assert_allclose(
            experiment.infer((x, v, y), uniform, a, s, 1),
            experiment.infer((x, v_masked, y_masked), uniform, a, s, 1),
        )

    # Rows of the controlled-study tables in the manuscript (main text and Supplementary Appendix).
    MANUSCRIPT_ROWS = {
        "main_table.tex": [
            r"Global & PCAL-grid & 22.27 (1.03) & 0.898 & 0.159 & 0.600 \\",
            r"Global & Plug-in & 22.04 (1.01) & 0.915 & 0.158 & 0.600 \\",
            r"Global & Gold-grid & 27.69 (1.18) & 0.886 & 0.173 & 0.600 \\",
            r"Global & Constant-grid & 22.78 (1.04) & 0.896 & 0.160 & 0.600 \\",
            r"Global & Oracle-gold & 27.96 (1.22) & 0.884 & 0.173 & 0.600 \\",
            r"None & PCAL-grid & 63.91 (3.01) & 0.903 & 0.269 & 0.600 \\",
            r"None & Plug-in & 64.43 (3.01) & 0.907 & 0.269 & 0.600 \\",
            r"None & Gold-grid & 64.03 (3.02) & 0.902 & 0.269 & 0.600 \\",
            r"None & Constant-grid & 63.61 (2.88) & 0.911 & 0.269 & 0.600 \\",
            r"None & Oracle-gold & 62.93 (2.82) & 0.912 & 0.267 & 0.600 \\",
            r"Local & PCAL-grid & 45.77 (1.97) & 0.909 & 0.227 & 0.600 \\",
            r"Local & Plug-in & 46.93 (2.09) & 0.903 & 0.225 & 0.600 \\",
            r"Local & Gold-grid & 50.51 (2.25) & 0.904 & 0.235 & 0.600 \\",
            r"Local & Constant-grid & 52.73 (2.39) & 0.904 & 0.243 & 0.600 \\",
            r"Local & Oracle-gold & 51.07 (2.29) & 0.898 & 0.234 & 0.600 \\",
        ],
        "supp_table.tex": [
            r"Global & PCAL-grid & -0.0001 & 0.010 & 0.175 & 0.125 & 23.05 \\",
            r"Global & Plug-in & -0.0001 & 0.009 & 0.209 & 0.161 & 22.91 \\",
            r"Global & Gold-grid & 0.0009 & 0.010 & 0.000 & 0.000 & 27.53 \\",
            r"Global & Constant-grid & 0.0004 & 0.010 & 0.150 & 0.150 & 23.33 \\",
            r"Global & Oracle-gold & 0.0006 & 0.010 & 0.000 & 0.000 & 27.44 \\",
            r"None & PCAL-grid & -0.0008 & 0.009 & 0.000 & 0.000 & 67.40 \\",
            r"None & Plug-in & -0.0002 & 0.009 & 0.000 & 0.000 & 67.20 \\",
            r"None & Gold-grid & -0.0008 & 0.009 & 0.000 & 0.000 & 67.32 \\",
            r"None & Constant-grid & -0.0020 & 0.009 & 0.000 & 0.000 & 67.50 \\",
            r"None & Oracle-gold & -0.0011 & 0.009 & 0.000 & 0.000 & 66.35 \\",
            r"Local & PCAL-grid & -0.0009 & 0.009 & 0.000 & 0.141 & 47.65 \\",
            r"Local & Plug-in & -0.0017 & 0.009 & 0.000 & 0.150 & 46.87 \\",
            r"Local & Gold-grid & -0.0007 & 0.009 & 0.000 & 0.000 & 51.30 \\",
            r"Local & Constant-grid & -0.0002 & 0.009 & 0.001 & 0.001 & 55.07 \\",
            r"Local & Oracle-gold & -0.0020 & 0.010 & 0.000 & 0.000 & 50.55 \\",
        ],
    }

    def test_manuscript_tables_match_generated_results(self):
        results = Path(__file__).resolve().parent / "controlled_results"
        for table, expected in self.MANUSCRIPT_ROWS.items():
            self.assertEqual((results / table).read_text().splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
