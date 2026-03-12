"""
attacks.py - Functions for testing fault detection against adversarial attacks

Design architecture:
--------------------
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
    Description
    -----------
    Function to convert target columns (target_cols) into a list of int column indices

    Accepted target_cols can be:
        - None         -> attack all columns
        - list of ints -> attack specified column indicies
    """
    if target_cols is None:
        return list(range(X_test.shape[1])) # Returns number of features in dataset as a list
    return list(target_cols)
    # NOTES: .shape returns dimensions as a tuple (rows,columns)



#========================== Bias injection ==========================
def bias_injection(X_test: np.ndarray, col_stds: np.ndarray, severity: float = 0.5, target_cols = None) -> np.ndarray:
    """
    Description
    -----------
    This attack aims to add a constant negative/malicious offset to selected columns of the testing data

    Offset is calculated by:
        offset = severity * col_std (where std refers to how much values can vary from the mean)

        severity = 0.5 -> offset is half the standard deviation (more subtle)
        severity = 2.0 -> Offset is double the standard deviations (obvious)
    
    Parameters
    ----------
    - X_test:       2D numpy array of shape (n_samples, n_features) representing the split test data
    - col_stds:     1D numpy array containing each columns standard deviation 
    - severity:     Scalar multiplier controlling offset magnitude, unitless as it is expressed in std
    - target_cols:  Which columns to corrupt (None specified = all columns, [0,1,2] = Ia, Ib, Ic)

    Returns:
    -------
    - Corrupted copy of X_test, origional is never modified
    """
    X_corrupted = X_test.copy() # Making sure to work on a copy of test data

    cols = resolve_target_cols(X_test, target_cols)

    for column_i in cols: # For each feature we want to corrupt
        offset = severity * col_stds[column_i] # create a fixed offset to be added to value in this column
        X_corrupted[:, column_i] += offset # For all rows (:) of column column_i add offset to each value

    return X_corrupted



#========================== Drift over time ==========================
def drift_over_time(X_test: np.ndarray, col_stds: np.ndarray, severity: float = 0.5, target_cols = None) -> np.ndarray:
    """
    Description
    -----------
    - A gradually increasing offset that grows with each sample index is applied
    cap is at the max corruption level

    How drift is calculated per sample i:
        rate     = severity * col_std / n_samples   <-- Slope of the ramp up
        cap      = severity * col_std               <-- ceiling
        drift[i] = min(rate * i, cap)

    - At i=0 (first sample): drift = 0 (no corruption yet)
    - At i=n (last sample):  drift = cap (fully degraded readings/samples)

    - dividing by n_samples ensures that the desired drift 'reaches' the cap at exactly the last sample
    regardless of how many samples there are in the test set. This keeps severity protable across
    different dataset sizes.

    Parameters
    ----------
    - X_test:      2D numpy array of shape (n_samples, n_features)
    - col_stds:    1D numpy array of per column stds from training data
    - severity:    Controls both the rate and final cap
    - target_cols: Which columns are being corrupted. None = all columns

    Returns:
    -------
    - A corrupted copy of X_test with time varying offsets applied
    """