"""
experiments/run_attacks.py

NEW PURPOSE:
------------
Generate corrupted datasets using adversarial attacks.
NO model evaluation is performed.

Outputs:
    data/attacks/*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Project root setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_detection_data, load_multiclass_data
from src.attacks.attacks import (
    bias_injection,
    drift_over_time,
    noise_injection,
    replay_attack
)


# ======================== ARGUMENTS ======================== #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate adversarial attack datasets (no evaluation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--detection",
        default=str(ROOT / "data" / "detect_dataset.csv"),
    )
    parser.add_argument(
        "--diagnostic",
        default=str(ROOT / "data" / "classData.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "attacks"),
        help="Directory to save generated CSV files"
    )

    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--target-cols",
        type=int,
        nargs="+",
        default=None
    )

    return parser.parse_args()


# ======================== MAIN ======================== #
def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  Attack Dataset Generation")
    print("=" * 65)
    print(f"Severity:    {args.severity}")
    print(f"Fraction:    {args.fraction}")
    print(f"Lookback:    {args.lookback}")
    print(f"Seed:        {args.seed}")
    print(f"Target cols: {args.target_cols}")
    print(f"Output dir:  {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================== LOAD DATA ======================== #
    print("\n[1/3] Loading data...")

    det_data = load_detection_data(args.detection, test_size=0.2, random_state=42)
    diag_data = load_multiclass_data(args.diagnostic, test_size=0.2, random_state=42)

    X_train_det = det_data["X_train"]
    X_test_det = det_data["X_test"]

    X_train_diag = diag_data["X_train"]
    X_test_diag = diag_data["X_test"]

    # Compute stds from training data
    col_stds_det = X_train_det.std(axis=0)
    col_stds_diag = X_train_diag.std(axis=0)

    print(f"Detection test shape:  {X_test_det.shape}")
    print(f"Diagnostic test shape: {X_test_diag.shape}")

    # Column names (assumes consistent ordering)
    columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

    # ======================== APPLY ATTACKS ======================== #
    print("\n[2/3] Applying attacks...")

    shared = dict(severity=args.severity, target_cols=args.target_cols)

    attacks = {
        "bias": lambda X, std: bias_injection(X, std, **shared),
        "drift": lambda X, std: drift_over_time(X, std, **shared),
        "noise": lambda X, std: noise_injection(
            X, std, fraction=args.fraction, seed=args.seed, **shared
        ),
        "replay": lambda X, std: replay_attack(
            X,
            std,
            severity=args.fraction,
            lookback=args.lookback,
            seed=args.seed,
            target_cols=args.target_cols,
        ),
    }

    # ======================== SAVE CSVs ======================== #
    print("\n[3/3] Saving CSV files...")

    for name, attack_fn in attacks.items():
        # Detection dataset
        X_corr_det = attack_fn(X_test_det, col_stds_det)
        df_det = pd.DataFrame(X_corr_det, columns=columns)

        det_path = output_dir / f"detection_{name}.csv"
        df_det.to_csv(det_path, index=False)

        # Diagnostic dataset
        X_corr_diag = attack_fn(X_test_diag, col_stds_diag)
        df_diag = pd.DataFrame(X_corr_diag, columns=columns)

        diag_path = output_dir / f"diagnostic_{name}.csv"
        df_diag.to_csv(diag_path, index=False)

        print(f"Saved: {det_path.name}, {diag_path.name}")

    print("\nAll attack datasets generated successfully.")


# ======================== ENTRY ======================== #
if __name__ == "__main__":
    main()