"""Reusable live logging for a black-box CMA-ES optimization."""

import argparse
import functools
import sys, os
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/experiments")

import evolution._lib import *

""" example """

class CurveObjective:
    """Example black-box objective used by ``mocksim``."""

    def __init__(self, num_parameters: int):
        x = np.linspace(0.0, 31.4, num_parameters)
        self.target = x
        self.m = np.random.randn(num_parameters, num_parameters)

    def __call__(self, population):
        population = np.asarray(population, dtype=float)
        fitness = []
        for ind in population:
            fitness += [np.mean((ind - self.target)**2)]

        return fitness
        # return np.mean((population - self.values) ** 2, axis=1)


def mocksim(num_parameters=64, generations=300, pause=0.01, live_plot=True):
    population_size = 4 + int(3 * np.log(num_parameters))
    settings = {
        "num_parameters": num_parameters,
        "generations": generations,
        "population_size": population_size,
        "pause_time": pause,
        "direction": "minimize",
    }
    return _lib.evolution_run(
        settings=settings,
        evaluate=CurveObjective(num_parameters),
        live_plot=live_plot,
    )

if __name__ == "__main__":
    mocksim()
