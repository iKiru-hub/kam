"""Live visualization of continual MTL learning on a two-cue circular track.

The dashboard updates online while the model traverses the track.  It shows
physical and decoded position, the CA1-decoded LEC pattern, and the changing
MEC/LEC preference of every CA1 neuron as CA3-to-CA1 plasticity proceeds.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments import track_analysis_experiment as analysis  # noqa: E402
from experiments import track_experiment as track  # noqa: E402


DEFAULT_FIGURE = Path(__file__).resolve().parent / "data" / \
    "track_live_experiment.png"
CUE_COLORS = ("tab:cyan", "tab:pink")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and visualize online learning on a two-cue track."
    )
    parser.add_argument("--ae-name", default="ae_cue_nb_7")
    parser.add_argument(
        "--seed", type=analysis.parse_seed, default=40,
        help="integer seed, or 'random' for a fresh reported seed",
    )
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--laps", type=int, default=20)
    parser.add_argument(
        "--swap-every", "--k", dest="swap_every", type=int, default=10,
        help="swap the two cue identities every k laps",
    )
    parser.add_argument("--lap-length", type=int, default=50)
    parser.add_argument(
        "--cue-positions", type=int, nargs=2, default=(10, 30),
        metavar=("CUE_1", "CUE_2"),
    )
    parser.add_argument(
        "--update-every", type=int, default=1,
        help="redraw after this many track samples",
    )
    parser.add_argument(
        "--pause", type=float, default=0.001,
        help="GUI pause in seconds after each redraw",
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--no-show", action="store_true",
        help="run headlessly and save only the final dashboard",
    )
    parser.add_argument(
        "--k-ca3", type=int, default=track.DEFAULT_MTL_SETTINGS["K_ca3"]
    )
    parser.add_argument(
        "--beta-ca3", type=float,
        default=track.DEFAULT_MTL_SETTINGS["beta_ca3"],
    )
    parser.add_argument(
        "--beta-ca1", type=float,
        default=track.DEFAULT_MTL_SETTINGS["beta_ca1"],
    )
    parser.add_argument(
        "--alpha", type=float, default=track.DEFAULT_MTL_SETTINGS["alpha"]
    )
    parser.add_argument(
        "--nb-ei-ca3", type=int,
        default=track.DEFAULT_MTL_SETTINGS["nb_ei_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca3", type=int,
        default=track.DEFAULT_MTL_SETTINGS["num_swaps_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca1", type=int,
        default=track.DEFAULT_MTL_SETTINGS["num_swaps_ca1"],
    )
    parser.add_argument(
        "--plasticity",
        choices=("base", "nois", "isout", "err1", "err2"),
        default=track.DEFAULT_MTL_SETTINGS["plasticity"],
    )
    return parser.parse_args()


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _modality_coordinates(weights: np.ndarray,
                          mec_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized MEC and LEC sensitivity for each CA1 neuron."""

    mec = np.linalg.norm(weights[:, :mec_size], axis=1)
    lec = np.linalg.norm(weights[:, mec_size:], axis=1)
    return _normalize(mec), _normalize(lec)


def _preferred_spatial_position(weights: np.ndarray, mec_size: int,
                                track_length: int) -> np.ndarray:
    """Decode each neuron's preferred position from its effective MEC drive."""

    mec_weights = np.abs(weights[:, :mec_size])
    unit_angles = 2.0 * np.pi * np.arange(mec_size) / mec_size
    vectors = mec_weights @ np.exp(1j * unit_angles)
    return (
        np.mod(np.angle(vectors), 2.0 * np.pi)
        * track_length / (2.0 * np.pi)
    )


def _cue_templates(stimuli: np.ndarray, assignments: list[list[int]],
                   cue_positions: list[int]) -> np.ndarray:
    mec_size = stimuli.shape[-1] // 2
    samples = {0: [], 1: []}
    for lap_index, lap_assignment in enumerate(assignments):
        for slot, identity in enumerate(lap_assignment):
            samples[int(identity)].append(
                stimuli[lap_index, cue_positions[slot], mec_size:]
            )
    return np.stack([
        np.mean(samples[identity], axis=0) for identity in (0, 1)
    ])


class LiveTrackFigure:
    """Efficient artist-based online dashboard for one track traversal."""

    def __init__(self, model, track_length: int, cue_positions: list[int],
                 cue_templates: np.ndarray, enabled: bool = True,
                 pause: float = 0.001):
        self.model = model
        self.track_length = int(track_length)
        self.cue_positions = np.asarray(cue_positions, dtype=int)
        self.cue_templates = np.asarray(cue_templates)
        self.mec_size = model._dim_ei // 2
        self.enabled = bool(enabled)
        self.pause = max(float(pause), 0.0)
        if self.enabled:
            plt.ion()

        self.figure = plt.figure(figsize=(15, 10), constrained_layout=True)
        grid = self.figure.add_gridspec(2, 2)
        self.track_axis = self.figure.add_subplot(grid[0, 0], projection="polar")
        self.decode_axis = self.figure.add_subplot(grid[0, 1], projection="polar")
        self.cue_axis = self.figure.add_subplot(grid[1, 0])
        self.preference_axis = self.figure.add_subplot(grid[1, 1])
        self._initialize_track_axes()
        self._initialize_cue_axis()
        self._initialize_preference_axis()

    def _position_angle(self, position) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(position) / self.track_length

    def _format_polar_track(self, axis, title: str):
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_ylim(0, 1.18)
        axis.set_yticks([])
        ticks = np.arange(0, self.track_length, 10)
        axis.set_xticks(self._position_angle(ticks))
        labels = [f"{tick} cm" for tick in ticks]
        labels[0] = f"0 / {self.track_length} cm"
        axis.set_xticklabels(labels)
        axis.set_title(title, pad=18)
        axis.grid(alpha=0.25)

    def _initialize_track_axes(self):
        self._format_polar_track(
            self.track_axis, "Physical position and current cue identities"
        )
        self._format_polar_track(
            self.decode_axis, "CA1-decoded spatial position"
        )
        self.current_position, = self.track_axis.plot(
            [0], [1], marker="o", color="black", markersize=9,
            linestyle="None", label="current position",
        )
        self.cue_markers = self.track_axis.scatter(
            self._position_angle(self.cue_positions), np.ones(2),
            s=180, marker="*", edgecolor="black", zorder=4,
        )
        self.cue_labels = [
            self.track_axis.text(
                self._position_angle(position), 1.10, "", ha="center",
                va="center", fontsize=9,
            )
            for position in self.cue_positions
        ]
        self.true_position, = self.decode_axis.plot(
            [0], [1], marker="o", color="black", markersize=8,
            linestyle="None", label="physical",
        )
        self.decoded_position, = self.decode_axis.plot(
            [0], [0.83], marker="D", color="tab:orange", markersize=8,
            linestyle="None", label="decoded from CA1→EO",
        )
        self.position_error_arc, = self.decode_axis.plot(
            [0, 0], [0.91, 0.91], color="tab:red", linewidth=2,
            alpha=0.75,
        )
        self.decode_axis.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.12))

    def _initialize_cue_axis(self):
        lec_size = self.model._dim_ei - self.mec_size
        cue_units = np.arange(lec_size)
        self.target_cue, = self.cue_axis.plot(
            cue_units, np.zeros(lec_size), color="0.35", linewidth=1.8,
            label="input LEC target",
        )
        self.decoded_cue, = self.cue_axis.plot(
            cue_units, np.zeros(lec_size), color="tab:purple", linewidth=2.0,
            label="decoded from CA1",
        )
        self.cue_axis.set(
            xlabel="LEC cue-pattern unit",
            ylabel="activity",
            xlim=(0, lec_size - 1),
            ylim=(-0.04, 1.04),
            title="CA1-decoded sensory cue pattern",
        )
        self.cue_axis.grid(alpha=0.18)
        self.cue_axis.legend(loc="upper right")

    def _initialize_preference_axis(self):
        instructive = self.model.W_ei_ca1.detach().cpu().numpy()
        anchor_mec, anchor_lec = _modality_coordinates(
            instructive, self.mec_size
        )
        self.preference_anchors = np.column_stack((anchor_mec, anchor_lec))
        self.preference_axis.scatter(
            anchor_mec, anchor_lec, marker="x", color="0.55", s=35,
            label="fixed instructive $W_{EI→CA1}$",
        )
        self.preference_links = LineCollection(
            [], colors="0.65", linewidths=0.6, alpha=0.35
        )
        self.preference_axis.add_collection(self.preference_links)
        self.preference_points = self.preference_axis.scatter(
            np.zeros(self.model._dim_ca1),
            np.zeros(self.model._dim_ca1),
            c=np.zeros(self.model._dim_ca1),
            cmap="twilight", vmin=0, vmax=self.track_length,
            s=np.full(self.model._dim_ca1, 25.0),
            edgecolor="black", linewidth=0.25,
            label="plastic effective $EI→CA3→CA1$",
        )
        self.preference_axis.set(
            xlabel="MEC sensitivity",
            ylabel="LEC sensitivity",
            xlim=(-0.06, 1.06),
            ylim=(-0.06, 1.06),
            title="CA1 modality preference and plastic displacement",
        )
        self.preference_axis.grid(alpha=0.18)
        self.preference_axis.legend(loc="upper right", fontsize=8)
        self.figure.colorbar(
            self.preference_points, ax=self.preference_axis,
            label="preferred circular-track position (cm)", shrink=0.82,
        )

    def _effective_weights(self) -> np.ndarray:
        ca3_ca1 = self.model.W_ca3_ca1.detach().cpu().numpy()
        ei_ca3 = self.model.W_ei_ca3.detach().cpu().numpy()
        return ca3_ca1 @ ei_ca3

    def update(self, lap_index: int, position: int, assignment: list[int],
               x_ei: np.ndarray, ca1: np.ndarray, eo: np.ndarray,
               force_draw: bool = False):
        physical_angle = float(self._position_angle(position))
        decoded = float(analysis._decode_circular_population(
            eo[None, :self.mec_size], self.track_length
        )[0])
        decoded_angle = float(self._position_angle(decoded))
        difference = (decoded - position + self.track_length / 2.0) \
            % self.track_length - self.track_length / 2.0

        self.current_position.set_data([physical_angle], [1.0])
        cue_colors = [CUE_COLORS[int(identity)] for identity in assignment]
        self.cue_markers.set_facecolors(cue_colors)
        for label, identity in zip(self.cue_labels, assignment):
            label.set_text(f"cue {identity}")
            label.set_color(CUE_COLORS[int(identity)])
        self.true_position.set_data([physical_angle], [1.0])
        self.decoded_position.set_data([decoded_angle], [0.83])
        arc_positions = np.linspace(position, position + difference, 40)
        self.position_error_arc.set_data(
            self._position_angle(arc_positions), np.full(40, 0.91)
        )
        self.decode_axis.set_title(
            "CA1-decoded spatial position\n"
            f"physical={position:.1f} cm, decoded={decoded:.1f} cm, "
            f"error={abs(difference):.1f} cm",
            pad=18,
        )

        target_lec = x_ei[self.mec_size:]
        decoded_lec = eo[self.mec_size:]
        self.target_cue.set_ydata(target_lec)
        self.decoded_cue.set_ydata(decoded_lec)
        similarities = (
            decoded_lec @ self.cue_templates.T
        ) / np.maximum(
            np.linalg.norm(decoded_lec)
            * np.linalg.norm(self.cue_templates, axis=1), 1e-8
        )
        predicted_cue = int(np.argmax(similarities))
        cue_present = float(np.max(decoded_lec)) >= 0.10
        prediction_label = str(predicted_cue) if cue_present else "none"
        self.cue_axis.set_title(
            "CA1-decoded sensory cue pattern\n"
            f"predicted cue={prediction_label}; similarities="
            f"[{similarities[0]:.2f}, {similarities[1]:.2f}]"
        )

        effective = self._effective_weights()
        dynamic_mec, dynamic_lec = _modality_coordinates(
            effective, self.mec_size
        )
        locations = np.column_stack((dynamic_mec, dynamic_lec))
        preferred = _preferred_spatial_position(
            effective, self.mec_size, self.track_length
        )
        activity = _normalize(ca1)
        self.preference_points.set_offsets(locations)
        self.preference_points.set_array(preferred)
        self.preference_points.set_sizes(20.0 + 115.0 * activity)
        self.preference_links.set_segments(np.stack(
            (self.preference_anchors, locations), axis=1
        ))
        self.preference_axis.set_title(
            "CA1 modality preference and plastic displacement\n"
            "color=preferred position; size=current CA1 activity"
        )
        self.figure.suptitle(
            f"Online MTL circular-track learning | lap {lap_index + 1}, "
            f"position {position}/{self.track_length - 1}, cues {assignment}"
        )

        if self.enabled or force_draw:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            if self.enabled:
                plt.pause(self.pause)

    def save(self, path: Path) -> Path:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=180)
        return path


def run_live_experiment(ae_name: str = "ae_cue_nb_7",
                        seed: int | None = 40,
                        num_laps: int = 20,
                        lap_length: int = 50,
                        swap_every: int = 10,
                        cue_positions=(10, 30),
                        update_every: int = 1,
                        pause: float = 0.001,
                        show: bool = True,
                        mtl_settings: dict | None = None):
    if num_laps < 1 or lap_length < 2 or swap_every < 1:
        raise ValueError("laps, lap length, and swap interval must be positive")
    if update_every < 1:
        raise ValueError("update_every must be at least 1")
    if len(cue_positions) != 2:
        raise ValueError("exactly two cue positions are required")
    if any(not 0 <= position < lap_length for position in cue_positions):
        raise ValueError("cue positions must lie within the track")
    resolved_seed = secrets.randbits(32) if seed is None else int(seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    data_settings = {
        **track.DEFAULT_DATA_SETTINGS,
        "num_laps": int(num_laps),
        "lap_length": int(lap_length),
        "swap_every": int(swap_every),
        "cue_positions": list(cue_positions),
    }
    stimuli, assignments = track.make_swapping_cue_track(data_settings)
    settings = {
        **track.DEFAULT_MTL_SETTINGS,
        **({} if mtl_settings is None else mtl_settings),
    }
    model, ae_session = track.build_mtl(ae_name, settings)
    templates = _cue_templates(stimuli, assignments, list(cue_positions))
    live = LiveTrackFigure(
        model, lap_length, list(cue_positions), templates,
        enabled=show, pause=pause,
    )

    sample_index = 0
    last_values = None
    for lap_index, lap in enumerate(stimuli):
        model.reset()
        model.resume_lr()
        for position, sample in enumerate(lap):
            x = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            with torch.no_grad():
                model(x)
            x_ei = track._vector(x)
            ca1 = track._vector(model._ca1)
            eo = track._vector(model._eo)
            last_values = (lap_index, position, assignments[lap_index],
                           x_ei, ca1, eo)
            if sample_index % update_every == 0:
                live.update(*last_values)
            sample_index += 1
    if last_values is not None:
        live.update(*last_values, force_draw=True)
    return {
        "model": model,
        "figure": live,
        "seed": resolved_seed,
        "seed_mode": "random" if seed is None else "fixed",
        "data_settings": data_settings,
        "mtl_settings": settings,
        "cue_assignments": assignments,
        "autoencoder_session": ae_session,
    }


def main():
    args = parse_args()
    mtl_settings = {
        **track.DEFAULT_MTL_SETTINGS,
        "K_ca3": args.k_ca3,
        "beta_ca3": args.beta_ca3,
        "beta_ca1": args.beta_ca1,
        "alpha": args.alpha,
        "nb_ei_ca3": args.nb_ei_ca3,
        "num_swaps_ca3": args.num_swaps_ca3,
        "num_swaps_ca1": args.num_swaps_ca1,
        "plasticity": args.plasticity,
    }
    result = run_live_experiment(
        ae_name=args.ae_name,
        seed=None if args.random_seed else args.seed,
        num_laps=args.laps,
        lap_length=args.lap_length,
        swap_every=args.swap_every,
        cue_positions=args.cue_positions,
        update_every=args.update_every,
        pause=args.pause,
        show=not args.no_show,
        mtl_settings=mtl_settings,
    )
    figure_path = result["figure"].save(args.figure)
    print(f"random seed: {result['seed']} ({result['seed_mode']})")
    print(f"final dashboard saved to {figure_path}")
    if not args.no_show:
        plt.ioff()
        plt.show()
    else:
        plt.close(result["figure"].figure)


if __name__ == "__main__":
    main()
