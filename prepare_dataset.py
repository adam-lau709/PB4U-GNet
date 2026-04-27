from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from utils.data_making import convert_vto_to_pkl

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PB4U_PROJECT", str(PROJECT_ROOT))
os.environ.setdefault("PB4U_DATA", str(PROJECT_ROOT / "data"))

from utils.defaults import DEFAULTS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VTO SMPL sequences into PB4U training format."
    )
    parser.add_argument(
        "--vto-dataset-path",
        type=Path,
        required=True,
        help="Path to the root of the cloned vto-dataset repository.",
    )
    parser.add_argument(
        "--garment",
        default="tshirt",
        help="Garment subfolder inside the VTO dataset to process (default: tshirt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    simulations_path = args.vto_dataset_path / args.garment / "simulations"
    if not simulations_path.exists():
        raise FileNotFoundError(
            f"Could not find simulations folder at: {simulations_path}"
        )

    out_root = Path(DEFAULTS.vto_root) / "smpl_parameters"
    out_root.mkdir(parents=True, exist_ok=True)

    simulation_paths = sorted(simulations_path.iterdir())
    print(f"Found {len(simulation_paths)} simulations in {simulations_path}")
    print(f"Saving converted sequences to {out_root}")

    for simulation_path in tqdm(simulation_paths):
        out_path = out_root / simulation_path.name
        convert_vto_to_pkl(simulation_path, out_path)

    print("Dataset preparation completed.")


if __name__ == "__main__":
    main()
