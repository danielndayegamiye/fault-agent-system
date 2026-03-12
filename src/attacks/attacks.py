"""
attacks.py - Functions for testing fault detection against adversarial attacks

Design architecture:
    - Corruption is applied only to the test data (X_test)
    - All functions return a corrupted copy of input dataset (origional never modified)
    - Stochastic attacks (noise, replay) accept a random seed for reproducibility
    - Severity is scaled to training data std so it is unitless and portable

Each function shares the same signature pattern:
    attack(X_test, col_stds, severity, target_cols, ...) -> np.ndarray
"""
import numpy as np


#========================== Shared helper function ==========================#
def resolve_target_cols(X_test: np.ndarray, target_cols) -> list[int]:
    """
    Function to convert target columns (target_cols) into a list of int column indices

    Accepted target_cols can be:
        - None         -> attack all columns
        - list of ints -> attack specified column indicies
    """
    if target_cols is None:
        return list(range(X_test.shape[1])) # Returns number of features in dataset as a list
    return list(target_cols)
    # NOTES: .shape returns dimensions as a tuple (rows,columns)