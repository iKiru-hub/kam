"""Train the paper's spatial+sensory autoencoder from one settings object.

Examples, from the repository root:

    python src/train_autoencoder.py
    python src/train_autoencoder.py --epochs 400 --lr 0.0005 --name ae_factorial_01
    python src/train_autoencoder.py --config my_settings.json
    python src/train_autoencoder.py --settings-json '{"seed": 9, "epochs": 20}' --no-save

The reusable Python API is ``run_autoencoder_experiment(settings_dict)`` in
``kamemory.autoencoder_experiment``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from kamemory.autoencoder_experiment import (
    load_settings_json,
    run_autoencoder_experiment,
)
from kamemory.io import PATHS


DEFAULT_CONFIG = PATHS.configs / "autoencoder_factorial.json"


def _merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--settings-json",
        default=None,
        help="inline JSON object merged over --config",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name", help="checkpoint directory name")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> dict:
    settings = load_settings_json(args.config)
    if args.settings_json:
        inline = json.loads(args.settings_json)
        if not isinstance(inline, dict):
            raise TypeError("--settings-json must decode to an object")
        settings = _merge(settings, inline)
    overrides: dict = {"training": {}, "save": {}}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.lr is not None:
        overrides["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.name is not None:
        overrides["save"]["name"] = args.name
    if args.output_directory is not None:
        overrides["save"]["directory"] = str(args.output_directory)
    if args.no_save:
        overrides["save"]["enabled"] = False
    return _merge(settings, overrides)


def main() -> int:
    settings = settings_from_args(parse_args())
    result = run_autoencoder_experiment(settings)
    report = result["report"]
    test = report["metrics"]["trained"]["test"]
    reference = report["metrics"].get("reference_checkpoint_test")
    print("\nfinal held-out metrics")
    print(json.dumps(test, indent=2, sort_keys=True))
    if reference is not None:
        print("\nreference checkpoint held-out metrics")
        print(json.dumps(reference, indent=2, sort_keys=True))
    if result["session_path"] is not None:
        print(f"\nsaved checkpoint: {result['session_path']}")
        print(
            "reload with: "
            f"load_autoencoder_session({str(result['session_path'])!r})"
        )
    return 0

# =============================================================================
# Local utils
# =============================================================================


def load_session(idx: int=None,
                 verbose: bool=True) -> tuple:

    """
    Load the training information and the autoencoder model from the saved
    sessions

    Parameters
    ----------
    idx : int
        the index of the session to load.
        Default is None.
    verbose : bool
        print the training information.
        Default is True.

    Returns
    -------
    info : dict
        training information
    model : object
        autoencoder model
    """

    # display the saved sessions
    try:
        ae_sessions = [f for f in os.listdir(utils.AE_PATH) if "ae" in f]
    except FileNotFoundError:
        logger.error(f"import error, maybe wrong path?\n" + \
                     f"current: {os.getcwd()}\nqueried: {utils.AE_PATH}")

    if len(ae_sessions) == 0:
        raise ValueError("No saved sessions found")

    # -- select autoencoder
    if idx is None or idx < 0:

        logger("Saved sessions:")
        for i, session in enumerate(ae_sessions):
            print(f"[{i}] {session}")

        # select the session
        idx = int(input("Select session\n>>> "))
    elif verbose:
        logger(f"Pre-selected session: [{idx}]")

    # -- load selected session
    session = ae_sessions[idx]
    with open(f"{utils.AE_PATH}/{session}/info.json", "r") as f:
        info = json.load(f)

    if "network_params" in info:
        input_dim = info["network_params"]["dim_ei"]
        encoding_dim = info["network_params"]["dim_ca1"]
        K = info["network_params"]["K_ca1"]
        beta = info["network_params"]["beta_ca1"]
        bias = info["network_params"]["bias"]
    else:
        input_dim = info["dim_ei"]
        encoding_dim = info["dim_ca1"]
        K = info["K_lat"]
        beta = info["beta"]
        try:
            bias = info["bias"]
        except KeyError:
            logger.warning("bias not found in the info file, set to True")
            bias = True

    # -- declare autoencoder
    model = Autoencoder(input_dim=input_dim,
                        encoding_dim=encoding_dim,
                        activation=None,
                        K=K,
                        beta=beta,
                        use_bias=bias)

    model.load_state_dict(torch.load(f"{utils.AE_PATH}/{session}/autoencoder.pt"))

    if verbose:
        logger("Retrieved autoencoder hyper-parameters and session:")
        pprint(info)

    return info, model


def local_main():

    """ local testing function """

    dim_ei = 100
    dim_ca3 = 200
    dim_ca1 = 150
    dim_eo = 100
    K_lat = 10
    K_out = 11
    K_ca3 = 11
    beta = 10.
    alpha=0.1

    model = MTL(W_ei_ca1=torch.randn(dim_ca1, dim_ei),
                W_ca1_eo=torch.randn(dim_eo, dim_ca1),
                K_lat=K_lat,
                K_out=K_out,
                K_ca3=K_ca3,
                beta=beta,
                alpha=alpha,
                dim_ca3=dim_ca3)

    input_data = torch.randn(dim_ei, 1)  # Batch size of 1 for simplicity

    with torch.no_grad():
        output_data = model(input_data)

    logger(f"input shape: {input_data.shape}")
    logger(f"input shape: {output_data.shape}")

    # Generate some random data
    N = 10
    size = 50

    heads = 3
    variance = 0.05
    higher_heads = heads 
    higher_variance = 0.075

    samples = utils.stimulus_generator(N, size, heads, variance,
                                       higher_heads=higher_heads,
                                       higher_variance=higher_variance,
                                       plot=False)

    # make model
    model = Autoencoder(input_dim=size, encoding_dim=10)

    # train model
    epochs = 2e3
    loss, model = training.train_autoencoder(samples, samples, model, epochs=int(epochs),
                                             batch_size=5, learning_rate=1e-3)
    logger(f"autoencoder loss: {loss:.3f}")

    # reconstruct data
    training.reconstruct_data(samples, num=5, model=model)



if __name__ == "__main__":
    raise SystemExit(main())
