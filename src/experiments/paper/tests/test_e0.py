from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.paper.figures import build
from experiments.paper.development import select_development_configuration
from experiments.paper.final_e1 import evaluate_gate_a
from experiments.paper.e2_retention import evaluate_gate_b1
from experiments.paper.metrics import metric_sanity_checks
from experiments.paper.runner import run
from experiments.paper.seeds import (
    SeedStreams,
    assert_seed_sets_disjoint,
    random_derangement,
    random_nonidentity_permutation,
)
from experiments.paper.validate import validate


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PACKAGE_ROOT / "configs" / "e0_smoke.json"


class SeedAndMetricTests(unittest.TestCase):
    def test_seed_sets_are_disjoint(self) -> None:
        assert_seed_sets_disjoint()

    def test_streams_are_order_independent(self) -> None:
        first = SeedStreams(123).numpy("memory_bank").integers(0, 100, size=10)
        streams = SeedStreams(123)
        streams.numpy("ca3_wiring").integers(0, 100, size=100)
        second = streams.numpy("memory_bank").integers(0, 100, size=10)
        np.testing.assert_array_equal(first, second)

    def test_permutation_and_derangement(self) -> None:
        rng = np.random.default_rng(4)
        permutation = random_nonidentity_permutation(12, rng)
        self.assertFalse(np.array_equal(permutation, np.arange(12)))
        derangement = random_derangement(12, rng)
        self.assertTrue(np.all(derangement != np.arange(12)))

    def test_metric_edge_cases(self) -> None:
        self.assertTrue(all(metric_sanity_checks().values()))

    def test_development_selection_prefers_base_and_one_se_rate(self) -> None:
        # rule × alpha × seed × condition × query
        values = np.zeros((2, 3, 8, 5, 2), dtype=np.float64)
        values[:, :, :, 1, :] = 0.30  # fixed
        values[:, :, :, 2, :] = 0.80  # rescue
        values[0, 0, :, 0, :] = 0.72
        values[0, 1, :, 0, :] = 0.80
        values[0, 2, :, 0, :] = 0.81
        values[1, :, :, 0, :] = 0.60  # ERR2 infeasible in this synthetic case
        selection = select_development_configuration(
            values,
            ["base", "err2"],
            np.asarray([0.01, 0.02, 0.04]),
            [
                "aligned",
                "fixed_permutation",
                "matched_decoder_rescue",
                "random_content_matched",
                "no_plasticity",
            ],
            {
                "minimum_positive_seeds": 7,
                "minimum_aligned_cosine": 0.70,
                "minimum_alignment_effect": 0.20,
                "rescue_equivalence_margin": 0.02,
                "equivalence_t_critical_df7": 1.894578605061305,
            },
        )
        self.assertEqual(selection["primary_rule"], "base")
        self.assertEqual(selection["primary_alpha"], 0.04)

    def test_gate_a_calculation(self) -> None:
        values = np.tile(np.asarray([0.85, 0.10, 0.85, 0.12, 0.08]), (20, 1))
        result = evaluate_gate_a(
            values,
            [
                "aligned",
                "fixed_permutation",
                "matched_decoder_rescue",
                "random_content_matched",
                "no_plasticity",
            ],
            {
                "superiority_t_critical_df19": 2.093024054408263,
                "equivalence_t_critical_df19": 1.729132811521367,
                "minimum_alignment_effect": 0.20,
                "rescue_equivalence_margin": 0.02,
            },
        )
        self.assertTrue(result["gate_a_pass"])

    def test_gate_b1_calculation(self) -> None:
        values = np.tile(np.asarray([0.75, 0.15, 0.75, 0.18, 0.12]), (20, 1))
        result = evaluate_gate_b1(
            values,
            [
                "aligned",
                "fixed_permutation",
                "matched_decoder_rescue",
                "random_content_matched",
                "no_plasticity",
            ],
            {
                "superiority_t_critical_df19": 2.093024054408263,
                "equivalence_t_critical_df19": 1.729132811521367,
                "rescue_equivalence_margin": 0.02,
            },
        )
        self.assertTrue(result["gate_b1_pass"])


class ArtifactTests(unittest.TestCase):
    def test_smoke_run_repeats_and_figure_uses_saved_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            report_a = run(SMOKE_CONFIG, first)
            report_b = run(SMOKE_CONFIG, second)
            self.assertEqual(report_a["scientific_digest"], report_b["scientific_digest"])
            self.assertTrue(all(validate(first).values()))
            figure = root / "figure.png"
            result = build(first, figure)
            self.assertTrue(figure.exists())
            self.assertTrue(figure.with_suffix(".csv").exists())
            self.assertEqual(result["scientific_digest"], report_a["scientific_digest"])


if __name__ == "__main__":
    unittest.main()
