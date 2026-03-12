"""
experiments/run_attacks.py
--------------------------
This script is used to stress test and evaluate pre trained agents by corrupting test 
data with four attack types and meassuring how much model performance degrades

How it connects to run_pipeline.py:
    run_pipeline.py  ->   trains agents, saves .pkl files to models/
    run_attacks.py   ->   Loads saved agents, atacks test data, reports accuracy and recall drop

usage:
    # From project rool
    python experiments/run_attacks.py

    # With custom parameters:
    python experiments/run_attacks.py \\
        --severity 0.5 \\
        --fraction 0.3 \\
        --loopback 50 \\
        -- seed 42

    # Attack only current sensors (Ia=0, Ib=1, Ic=2):
    python experiments/run_attacks.py --target-cols 0 1 2

    # Attack only voltage sensors (Va=3, Vb=4, Vc=5):
    python experiments/run_attacks.py --target-cols 3 4 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path(__file__) is this file: experiments/run_attacks.py
# .resolve().parent:           experiments/
# .parent                      project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.detection_agent import DetectionAgent
from src.agents.diagnostic_agent import DiagnosticAgent
from src.data.loader import load_detection_data, load_multiclass_data
from src.attacks.attacks import (bias_injection, drift_over_time, noise_injection, replay_attack)

#=================================================
# This will be an import for the evaluator and results
# DO NOT FORGET TO MAKE THIS AND ADD THIS
#=====================================================

def parse_args():
    """
    Define and parse all command line arguements. Kept in seperate function
    so it can be imported without triggering full pipeline
    """
    parser = argparse.ArgumentParser(
        description="Adversarial attack evaluation for fault detection agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data and model paths
    parser.add_arguement(
        "--detection",
        default=str(ROOT / "data" / "detect_dataset.csv"),
        help="Path to the bianry detection dataset CSV."
    )
    parser.add_arguement(
        "--diagnostic",
        default=str(ROOT / "data" / "detect_dataset.csv"),
        help="Path to the binary detection dataset CSV."
    )
    parser.add_arguement(
        "--models-dir",
        default=str(ROOT / "models"),
        help="Directory containing detection_agent.pkl and diagnostic_agent.pkl"    
    )
    
    # Sahred attack severity
    parser.add_arguement(
        "--severity",
        type=float,
        default=0.5,
        help=(
            "Attack severity expressed in units of per column standard deviation."
            "Applied uniformly to all four attacks"
            "0.1 subtle   0.5=moderate   1.0=strong   2.0=extreme"
        )
    )

    # Attack specific parameters
    parser.add_arguement(
        "--fraction",
        type=float,
        default=0.3,
        help=(
            "Fraction of test samples to corrupt for noise and replay attacks. "
            "ranges from 0.0 to 1.0   0.1=sparse attack   0.8=widespread attack."
        )
    )
    parser.add_arguement(
        "--lookback",
        type=int,
        default=50,
        help=(
            "Max samples to reference for replay attack. "
            "Small values (5-10) will replay recent data and be more subtle."
            "Large values (100-500) will replay distant past (more dangerous)."
        )
    )
    parser.add_arguement(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for attack involving randomness (noise + replay)." \
            "Same seed = identical results across run. "
            "Try multiple seed to verify findings are not seed dependent."
        )  
    )

    # target column
    parser.add_arguement(
        "--target-cols",
        type=int,
        nargs="+",
        default=None,
        help=(
            "column indices to attack (0-indexed 0:Ia 1=Ib 2=Ic 3=Va 4=Vb 5=Vc)."
            "Default (None) will attack all six columns."
            "Pass '--target-cols 0 1 2' for current sensors only"
            "Pass '--target-cols 3 4 5' for voltage sensors only"
        )
    )

    return parser.parse_args()



#======================== Main ========================
def main():
    args = parse_args()