from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.preprint import compatibility, completion, cue_remapping, cue_swap_control
from experiments.preprint.artifacts import load_arrays
from experiments.preprint.figures import (
    figure_2_compatibility,
    figure_3_cue_remapping,
    figure_4_completion,
    figure_cue_swap_control,
)


def config() -> dict:
    return {
        "root_seeds": [991, 992],
        "data": {"dimension": 12, "active": 3, "training_size": 32, "validation_size": 12, "memory_count": 6},
        "autoencoder": {"latent_dimension": 12, "beta_latent": 15.0, "beta_output": 15.0, "epochs": 4, "batch_size": 8, "learning_rate": 0.003},
        "memory": {"ca3_dimension": 12, "ca3_inputs_per_unit": 2, "k_ca3": 3, "k_ca1": 3, "beta_ca3": 40.0, "beta_ca1": 15.0, "beta_output": 15.0, "alpha": 0.08, "plasticity_rule": "base"},
        "track": {"training_laps": 4, "validation_laps": 2, "lap_length": 12, "size": 12, "cue_positions": [2, 8], "swap_every": 2, "cue_sigma": 2.0, "cue_beta": 10.0, "cue_alpha": 0.1, "mec_binarized": True, "mec_sigma": 2.0, "lec_sigma": 2.0},
        "completion": {"fractions": [0.0, 0.5], "masks_per_fraction": 2, "lec_only": False, "key_modes": ["normal", "shuffled", "identity"]},
    }


class SmokeTests(unittest.TestCase):
    def test_compatibility_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = compatibility.run(config(), Path(temporary) / "first")
            second = compatibility.run(config(), Path(temporary) / "second")
            self.assertEqual(load_arrays(first)["cosine"].tobytes(), load_arrays(second)["cosine"].tobytes())

    def test_track_and_completion_write_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            track = cue_remapping.run(config(), Path(temporary) / "track")
            complete = completion.run(config(), Path(temporary) / "completion")
            self.assertEqual(load_arrays(track)["probe_ca1"].shape[0], 2)
            self.assertEqual(load_arrays(complete)["output_cosine"].shape[:2], (2, 3))

    def test_cue_swap_control_is_matched_before_first_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = cue_swap_control.run(
                config(), Path(temporary) / "cue_swap_control"
            )
            arrays = load_arrays(artifact)
            self.assertEqual(arrays["transition_similarity"].shape, (2, 2, 3))
            self.assertTrue(np.array_equal(
                arrays["scheduled_ca1"][:, 0, :2],
                arrays["scheduled_ca1"][:, 1, :2],
            ))

    def test_figures_rebuild_from_saved_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = compatibility.run(config(), root / "compatibility")
            track = cue_remapping.run(config(), root / "track")
            complete = completion.run(config(), root / "completion")
            control = cue_swap_control.run(config(), root / "cue_swap_control")
            outputs = (
                figure_2_compatibility.build(first, root / "figure2.png"),
                figure_3_cue_remapping.build(track, root / "figure3.png"),
                figure_4_completion.build(complete, root / "figure4.png"),
                figure_cue_swap_control.build(control, root / "cue_swap.png"),
            )
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
