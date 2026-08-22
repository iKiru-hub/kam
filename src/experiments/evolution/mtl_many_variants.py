"""Run independent CMA-ES searches for multiple MTL plasticity variants."""

import argparse
import os
import sys

PROJECT_SRC = os.path.abspath(__file__).split("src")[0] + "src"
sys.path.append(PROJECT_SRC)
sys.path.append(PROJECT_SRC + "/experiments")

import mtl_evolution_experiment as mee
from core.logger import logger


DEFAULT_VARIANTS = mee.PLASTICITY_VARIANTS

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve MTL parameters independently for plasticity variants."
    )
    parser.add_argument("--generations", type=int, default=128)
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="workers per search; defaults to min(population size, CPU count)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        type=str.lower,
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def run_variants(generations=128, pause=0.1, live_plot=False,
                 workers=None, variants=DEFAULT_VARIANTS):
    """Run one complete, sequential search per requested plasticity rule."""
    records = {}
    for plasticity in variants:
        plasticity = str(plasticity).lower()
        logger(f"starting plasticity variant {plasticity.upper()}")
        res = mee.mtlsearch(
            generations=generations,
            pause=pause,
            live_plot=live_plot,
            workers=workers,
            plasticity=plasticity,
            verbose=False,
            save_name=f"mtl_evolution_{plasticity}",
        )
        records[plasticity] = res
        logger(f"best={res['best_fitness'][-1]:.4f}")
    return records


def main():
    args = parse_args()
    run_variants(
        generations=args.generations,
        pause=args.pause,
        live_plot=args.plot,
        workers=args.workers,
        variants=args.variants,
    )
    logger("all requested plasticity variants completed")


if __name__ == "__main__":
    main()
