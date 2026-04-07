"""
src/attacks/attacks.py - Functions for testing fault detection against adversarial attacks

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
import pandas as pd


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
    X_corrupted = X_test.copy()
    num_samples = X_test.shape[0]
    columns = resolve_target_cols(X_test, target_cols)

    # Drift vector with one value per sample, shape is (n_samples, )
    sample_indices = np.arange(num_samples) # [0, 1, 2, ..., n_samples-1]

    # For each column:
        # Compute that columns drift
        # Apply it to every row in that column

    for column_i in columns:
        cap = severity * col_stds[column_i]
        rate = cap / num_samples # Cap will be reached at the final samples

        drift_vector = np.minimum(rate * sample_indices, cap)

        X_corrupted[:, column_i] += drift_vector# Apply the drift to every row in column

    return X_corrupted



#========================== Noise injection ==========================
def noise_injection(X_test: np.ndarray, col_stds: np.ndarray, severity: float = 0.5, fraction: float = 0.3, target_cols = None, seed: int = 42 ) -> np.ndarray:
    """
    Description
    -----------
    - Function that adds normally distributed random noise to a random subset of samples

    Two independent severity parameters
    ----------------------------------
    - Severity -> std of the noise (spread of corruption)
    - Fraction -> proportion of samples being corrupted (0.0 to 1.0)
    
    Parameters
    ----------
    - X_test:       2D np array of shape (n_samples, n_features)
    - col_stds:     1D np array of columns stds from training data
    - severity:     std of the Gaussian noise, in units of col_std
    - fraction:     proportion of sampes being corrupted must be in range [0.0, 1.0]
    - target_cols:  Which columns are being corrupted where None = all columns
    - seed:         Random seed used for reproducibility. Same seed gives same results

    Returns
    -------
    - A corrupted copy of X_test
    """
    X_corrupted = X_test.copy() # Make a copy of the test data
    n_samples = X_test.shape[0]
    columns = resolve_target_cols(X_test, target_cols)# Obtain indicies of target columns


    rng = np.random.default_rng(seed)# Seed the RNG once before all operations
    # Ensures taht the same fraction of samples and same noise value are selected
    # every time the function is called with the same seed

    n_corrupted = int(n_samples * fraction) # Choose sample indices that will be corrupted
    corrupted_indicies = rng.choice(n_samples, size=n_corrupted, replace = False)
    # chooses n_corrupted sammples without replacement, no sample is corrupted twice

    for column_i in columns:
        noise_std = severity * col_stds[column_i]

        # Create noise only for the chosen corruped samples
        noise = rng.normal(loc=0.0, scale=noise_std, size=n_corrupted)
        X_corrupted[corrupted_indicies, column_i] += noise

    return X_corrupted 



#========================== Replay attack ==========================
def replay_attack(X_test: np.ndarray, col_stds: np.ndarray, severity: float = 0.5, lookback: int = 50, target_cols = None, seed: int = 42) -> np.ndarray:
    """
    Description
    -----------
    - Function to replace samples with older recorded data from the same test sequence

    How it works
    ------------
    1. Randomly select a fraction of sample indices to replace
    2. For each selected index, repalce with data from index (i - offset)
    3. If (i-offset) < 0, skip the sample since that implie looking from before the start


    Parameters
    ----------
    - X_test:       2D numpy array of shape (n_samples, n_features)
    - col_stds:     1D array of column stds from training data, not used for scaling but for consistent API
    - severity:     Fraction of samples to replace, value between [0.0, 1.0]
    - lookback:     Maximum number of sampels to go back to for replay
    - target_cols:  Which columns to corrupt where None = all columns
    - seed:         Random seed for reproducibility

    Returns
    -------
    - A corrupted copy of X_copy
    """
    X_corrupted = X_test.copy()
    n_samples = X_test.shape[0]
    cols = resolve_target_cols(X_test, target_cols)

    rng = np.random.default_rng(seed)

    # Calculate how many samples to replace
    n_replayed = int(n_samples * severity)
    target_indices = rng.choice(n_samples, size=n_replayed, replace=False)

    for target_index in target_indices:
        # picks a random offset between 1 and loopback
        offset = rng.integers(low=1, high=lookback + 1)
        source_index = target_index - offset  

        if source_index < 0:# if source index would be before start of array
            continue # Skip

        for column_i in cols: # Replace target rows selected coluns with source row
            X_corrupted[target_index, column_i] = X_test[source_index, column_i]

    return X_corrupted