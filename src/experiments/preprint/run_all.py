"""Run the three frozen final simulations into one results directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.preprint import compatibility, completion, cue_remapping
from experiments.preprint.config import read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path(__file__).with_name("configs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = (("compatibility", compatibility.run), ("cue_remapping", cue_remapping.run), ("completion", completion.run))
    for name, function in jobs:
        config = read_config(args.configs / f"final_{name}.json")
        print(function(config, args.output / name))


if __name__ == "__main__":
    main()
