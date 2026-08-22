import argparse
import sys, os
import json
import multiprocessing as mp
import warnings
from collections.abc import Callable
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp1" / "build"))
import evolution

DATA_PATH = "/logs/"


def _initialize_evaluation_worker():
    """Keep each process single-threaded when the population is parallelized."""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass



class LivePlot:
    """Render a generic optimization record without knowing the objective.

    The instance is callable and expects a record containing these lists:
    ``generations``, ``populations``, ``fitness``, ``optimizer_means``,
    ``best_candidates``, ``generation_best``, ``best_fitness``, and
    ``population_mean_fitness``. An optional ``sigma`` list is also plotted.

    Every list must contain one entry per completed generation. This makes the
    class independent of the evaluator and mostly independent of the optimizer.
    """

    REQUIRED_FIELDS = (
        "generations",
        "populations",
        "fitness",
        "optimizer_means",
        "best_candidates",
        "generation_best",
        "best_fitness",
        "population_mean_fitness",
    )

    def __init__(self, pause=0.01, fitness_label="fitness"):
        self.pause = pause
        self.fitness_label = fitness_label
        self.initialized = False
        self.figure = plt.figure(figsize=(13, 8), constrained_layout=True)
        grid = self.figure.add_gridspec(2, 2, height_ratios=(1.3, 1.0))
        self.candidate_ax = self.figure.add_subplot(grid[0, :])
        self.fitness_ax = self.figure.add_subplot(grid[1, 0])
        self.history_ax = self.figure.add_subplot(grid[1, 1])
        self.sigma_ax = None
        plt.ion()

    def _validate_record(self, record):
        missing = [field for field in self.REQUIRED_FIELDS if field not in record]
        if missing:
            raise KeyError(f"optimization record is missing: {', '.join(missing)}")

        length = len(record["generations"])
        if length == 0:
            raise ValueError("optimization record is empty")
        inconsistent = [
            field for field in self.REQUIRED_FIELDS
            if len(record[field]) != length
        ]
        if inconsistent:
            raise ValueError(
                "record fields have inconsistent lengths: "
                + ", ".join(inconsistent)
            )

    def _initialize(self, record):
        population = np.asarray(record["populations"][-1], dtype=float)
        if population.ndim != 2:
            raise ValueError("each population must be a 2D candidate matrix")

        parameter_axis = np.arange(population.shape[1])
        self.population_lines = [
            self.candidate_ax.plot(
                parameter_axis,
                candidate,
                color="tab:blue",
                alpha=0.12,
                linewidth=0.8,
            )[0]
            for candidate in population
        ]
        self.mean_line, = self.candidate_ax.plot(
            [], [], color="tab:orange", linewidth=2.0, label="optimizer mean"
        )
        self.best_candidate_line, = self.candidate_ax.plot(
            [], [], color="tab:green", linewidth=2.2, label="best candidate seen"
        )
        self.candidate_ax.set(
            xlim=(parameter_axis[0], parameter_axis[-1]),
            xlabel="parameter index",
            ylabel="parameter value",
            title="Current search population",
        )
        self.candidate_ax.grid(alpha=0.2)
        self.candidate_ax.legend(loc="upper right")

        self.generation_best_line, = self.fitness_ax.plot(
            [], [], color="0.65", label="generation best"
        )
        self.best_fitness_line, = self.fitness_ax.plot(
            [], [], color="tab:green", linewidth=2.0, label="best seen"
        )
        self.population_mean_line, = self.fitness_ax.plot(
            [], [], color="tab:blue", alpha=0.8, label="population mean"
        )
        self.fitness_ax.set(
            ylim=((0., 1.)),
            xlabel="generation",
            ylabel=self.fitness_label,
            title=f"{self.fitness_label.replace('_', ' ').title()} history",
            # yscale="symlog",
        )
        self.fitness_ax.grid(True, which="both", alpha=0.25)

        if "sigma" in record:
            self.sigma_ax = self.fitness_ax.twinx()
            self.sigma_line, = self.sigma_ax.plot(
                [], [], color="tab:purple", linestyle=":", label="sigma"
            )
            self.sigma_ax.set_ylabel("search scale", color="tab:purple")
            handles = [
                self.generation_best_line,
                self.best_fitness_line,
                self.population_mean_line,
                self.sigma_line,
            ]
        else:
            handles = [
                self.generation_best_line,
                self.best_fitness_line,
                self.population_mean_line,
            ]
        self.fitness_ax.legend(handles=handles, loc="best")

        initial_history = np.zeros((1, population.shape[1]))
        self.history_image = self.history_ax.imshow(
            initial_history,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            extent=(parameter_axis[0], parameter_axis[-1], 0, 1),
        )
        self.history_ax.set(
            xlabel="parameter index",
            ylabel="generation",
            title="Optimizer mean through time",
        )
        colorbar = self.figure.colorbar(self.history_image, ax=self.history_ax)
        colorbar.set_label("parameter value")
        self.initialized = True

    def __call__(self, record):
        self._validate_record(record)
        if not self.initialized:
            self._initialize(record)

        generations = np.asarray(record["generations"])
        population = np.asarray(record["populations"][-1], dtype=float)
        optimizer_mean = np.asarray(record["optimizer_means"][-1], dtype=float)
        best_candidate = np.asarray(record["best_candidates"][-1], dtype=float)
        parameter_axis = np.arange(population.shape[1])

        if len(population) != len(self.population_lines):
            raise ValueError("population size changed after LivePlot initialization")
        for line, candidate in zip(self.population_lines, population):
            line.set_data(parameter_axis, candidate)
        self.mean_line.set_data(parameter_axis, optimizer_mean)
        self.best_candidate_line.set_data(parameter_axis, best_candidate)
        self.candidate_ax.relim()
        self.candidate_ax.autoscale_view(scalex=False, scaley=True)
        self.candidate_ax.set_title(
            f"Generation {generations[-1]} | best {self.fitness_label} "
            f"{record['best_fitness'][-1]:.4f}"
        )

        self.generation_best_line.set_data(generations, record["generation_best"])
        self.best_fitness_line.set_data(generations, record["best_fitness"])
        self.population_mean_line.set_data(
            generations, record["population_mean_fitness"]
        )
        # self.fitness_ax.relim()
        # self.fitness_ax.autoscale_view()
        self.fitness_ax.legend(loc="lower left")

        if self.sigma_ax is not None:
            self.sigma_line.set_data(generations, record["sigma"])
            self.sigma_ax.relim()
            self.sigma_ax.autoscale_view()

        mean_history = np.asarray(record["optimizer_means"], dtype=float)
        self.history_image.set_data(mean_history)
        self.history_image.set_extent((
            parameter_axis[0], parameter_axis[-1], 0, len(mean_history)
        ))
        self.history_ax.set_ylim(0, len(mean_history))
        finite_values = mean_history[np.isfinite(mean_history)]
        if finite_values.size:
            limit = max(float(np.percentile(np.abs(finite_values), 98)), 1e-9)
            self.history_image.set_clim(-limit, limit)

        self.figure.canvas.draw_idle()
        plt.pause(self.pause)

    def show(self):
        plt.ioff()
        plt.show()


def make_record(direction="minimize", metric_name="fitness"):
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be 'minimize' or 'maximize'")
    return {
        "direction": direction,
        "metric_name": metric_name,
        "generations": [],
        "populations": [],
        "fitness": [],
        "optimizer_means": [],
        "best_candidates": [],
        "generation_best": [],
        "best_fitness": [],
        "population_mean_fitness": [],
        "sigma": [],
        "raw_populations": [],
        "raw_optimizer_means": [],
        "raw_best_candidates": [],
    }

def evolution_run(settings: dict, evaluate: Callable,
                  sanitizer: Callable|None=None,
                  evaluate_individual: Callable|None=None,
                  disable: bool=False,
                  live_plot=True):
    """Run CMA-ES, optionally evaluating population members in worker processes.

    ``evaluate`` is the batch/sequential evaluator. When the picklable
    ``evaluate_individual`` callable is supplied, ``settings["workers"]``
    controls a persistent process pool; by default one worker is used per
    population member, capped by the available CPU count.
    """
    num_parameters = settings["num_parameters"]
    generations = settings["generations"]
    population_size = settings["population_size"]
    direction = settings.get("direction", "minimize")
    verbose = settings.get("verbose", True)
    disable = settings.get("disable", False)
    if verbose: disable = True
    metric_name = settings.get("metric_name", "fitness")
    requested_workers = settings.get("workers")
    available_cpus = os.cpu_count() or 1
    workers = min(population_size, available_cpus) if requested_workers is None \
        else int(requested_workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    workers = min(workers, population_size, available_cpus)
    use_multiprocessing = workers > 1 and evaluate_individual is not None

    if sanitizer is None:
        sanitizer = lambda x: np.asarray(x, dtype=float)

    # The current C++ CMAES implementation always minimizes. Maximization is
    # supported at this template boundary by negating values before update.
    minimize = direction == "minimize"
    record = make_record(direction, metric_name)
    record["workers"] = workers if use_multiprocessing else 1
    optimizer = evolution.CMAES(num_parameters)
    live = LivePlot(
        settings.get("pause_time", 0.01),
        fitness_label=metric_name,
    ) if live_plot else None

    best_candidate = None
    best_raw_candidate = None
    best_fitness = float("inf") if minimize else -float("inf")

    pool = None
    if use_multiprocessing:
        start_method = settings.get("multiprocessing_start_method", "spawn")
        # Set these before spawning too, so native libraries imported while a
        # child starts do not create their own thread pools. Restore the
        # parent's environment once all children have inherited the values.
        thread_variables = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        previous_thread_settings = {
            variable: os.environ.get(variable)
            for variable in thread_variables
        }
        try:
            for variable in thread_variables:
                os.environ[variable] = "1"
            pool = mp.get_context(start_method).Pool(
                processes=workers,
                initializer=_initialize_evaluation_worker,
            )
        finally:
            for variable, value in previous_thread_settings.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value
        if verbose:
            print(
                f"parallel_evaluation workers={workers} "
                f"start_method={start_method}"
            )

    try:
        for generation in tqdm(range(generations), disable=disable):

            # CMA-ES must be updated with the exact latent samples it generated.
            # Decoding/clipping is used only for evaluator-facing parameters.
            raw_population = np.asarray(
                [optimizer.sample() for _ in range(population_size)],
                dtype=float,
            )
            population = np.asarray(
                [sanitizer(candidate.copy()) for candidate in raw_population],
                dtype=float,
            )
            if population.shape != raw_population.shape:
                raise ValueError(
                    "sanitizer must return one decoded value per latent parameter; "
                    f"expected {raw_population.shape}, got {population.shape}"
                )

            # Evaluate one candidate per worker. The batch evaluator remains as
            # the sequential fallback and for backwards-compatible objectives.
            if pool is not None:
                fitness_values = pool.map(
                    evaluate_individual,
                    [candidate.copy() for candidate in population],
                    chunksize=1,
                )
            else:
                fitness_values = evaluate(population)
            fitness = np.asarray(fitness_values, dtype=float)
            if fitness.shape != (population_size,):
                raise ValueError(
                    "evaluation must return one scalar per candidate; "
                    f"expected {(population_size,)}, got {fitness.shape}"
                )
            if not np.all(np.isfinite(fitness)):
                raise ValueError("evaluation returned non-finite fitness")

            # - update
            optimizer_fitness = fitness if minimize else -fitness
            optimizer.update(raw_population.tolist(), optimizer_fitness.tolist())

            # - logs
            generation_best_index = (
                int(np.argmin(fitness)) if minimize else int(np.argmax(fitness))
            )
            generation_best = float(fitness[generation_best_index])
            improved = (
                generation_best < best_fitness if minimize
                else generation_best > best_fitness
            )
            if improved:
                best_fitness = generation_best
                best_candidate = population[generation_best_index].copy()
                best_raw_candidate = raw_population[generation_best_index].copy()

            raw_optimizer_mean = np.asarray(optimizer.mean, dtype=float)
            optimizer_mean = np.asarray(
                sanitizer(raw_optimizer_mean.copy()),
                dtype=float,
            )

            record["generations"].append(generation)
            record["populations"].append(population.copy())
            record["fitness"].append(fitness.copy())
            record["optimizer_means"].append(optimizer_mean.copy())
            record["best_candidates"].append(best_candidate.copy())
            record["generation_best"].append(generation_best)
            record["best_fitness"].append(best_fitness)
            record["population_mean_fitness"].append(float(np.mean(fitness)))
            record["sigma"].append(float(optimizer.sigma))
            record["raw_populations"].append(raw_population.copy())
            record["raw_optimizer_means"].append(raw_optimizer_mean.copy())
            record["raw_best_candidates"].append(best_raw_candidate.copy())

            if (generation % 5 == 0 or generation == generations - 1) and verbose:
                print(
                    f"generation={generation:4d}  "
                    f"best_{metric_name}={best_fitness:.4f}  "
                    f"population_mean_{metric_name}={np.mean(fitness):.4f}  "
                    f"sigma={optimizer.sigma:.4f}"
                )
            if live is not None:
                live(record)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    if live is not None:
        live.show()
    return record



""" functions """

def exp_eval(data: np.ndarray, sigma: float):

    """
    evaluation of the results as weighted average
    with explonential weights

    Parameters
    ----------
    data: np.ndarray
        shape (num_stimuli, num_stimuli)
    sigma: float
        standard deviation of the exponential kernel

    Return
    ------
    float
    """

    n = len(data)
    out = np.zeros(n)
    for r in range(n):
        denom = 0.
        for c in range(r, -1, -1):
            w = np.exp(-0.5*((r-c)/sigma)**2)
            out[r] += w * np.clip(data[r, c], 0., 1.)
            denom += w

        out[r] = out[r] / denom

    return out


def save_genome(info: dict, name: str):

    """ save an evolved genome """

    path = os.path.abspath(__file__).split("src")[0] + \
        "src/experiments/evolution/data"
    with open(f"{path}/{name}.json", "w") as f:
        json.dump(info, f)


def load_genome(index: int):

    """ load an evolved and saved genome """

    name = ""
    path = os.path.abspath(__file__).split("src")[0] + \
        "src/experiments/evolution/data"
    for i, f in enumerate(os.listdir(path)):
        if int(f.split('_')[1]) == index:
            name = f"{path}/{f}"
    if f == "":
        warnings.warn("index not found")
        return None

    with open(name, "r") as f:
        file = json.load(f)

    return file
