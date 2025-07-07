from config import *


####################################################################################################
# Synergies extraction and reconstruction fucntions
####################################################################################################



####################################################################################################
# PCA
####################################################################################################

def pca_emg(emg_data, n_components, scale_W=False, random_state=None, svd_solver='full'):
    """
    Applies Principal Component Analysis (PCA) to EMG data for dimensionality reduction 
    and synergy extraction.

    Args:
        emg_data (ndarray): Input EMG data matrix of shape (n_samples, n_muscles).
        n_components (int): Number of principal components (synergies) to extract.
        scale_W (bool): If True, scale scores (U) by the explained variance ratio.
        random_state (int or None): Random seed for reproducibility.
        svd_solver (str): SVD solver to use. Default is 'full' (LAPACK-based).

    Returns:
        H (ndarray): Principal components (muscle synergies), shape (n_components, n_muscles).
        W (ndarray): Projection of data onto components (temporal activations), shape (n_samples, n_components).
        mean (ndarray): Mean of the original data used for reconstruction.
        X_reconstructed (ndarray): Reconstructed EMG data using the selected principal components.
    """

    print("\nApplying PCA...")

    #preprocessing data for a clear reconstruction (centering and normalization)
    X_centered = emg_data - np.mean(emg_data, axis=0)
    X_normalized = (emg_data - np.mean(emg_data, axis=0)) / np.std(emg_data, axis=0)

    #model pca
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    
    #extracting matrices
    W = pca.fit_transform(emg_data)             # Synergies over time
    H = pca.components_                         # Muscle patterns (Muscular synergy matrix)
    mean = pca.mean_
    
    # Scale scores by explained variance (makes them more comparable)
    if scale_W:
        W = W * np.sqrt(pca.explained_variance_ratio_)
    # Transpose to keep same structure as NMF function
    if H.shape[0] != n_components:
        H = H.T     # Ensure S_m has shape (n_synergies, n_muscles)
    
    #reconstruction based on the inverse transform
    X_transformed = pca.fit_transform(X_centered) # Neural matrix (synergies over time) adjusted for centering wrt original data and enforce positive values
    X_reconstructed = pca.inverse_transform(X_transformed) + np.mean(emg_data, axis=0) # the mean is added to enforce values of synergies and reconstruction being non negative as the original data

    """mse = np.mean((emg_data - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse}")"""

    print("PCA completed.\n")

    return H, W, mean, X_reconstructed


#---------------------------------------------------------------------------------------------
def pca_emg_reconstruction(W, H, mean, n_synergies):
    """
    Reconstructs EMG data using a selected number of PCA components.

    Args:
        W (ndarray): Scores matrix (temporal activations), shape (n_samples, total_components).
        H (ndarray): Principal components (muscle synergies), shape (total_components, n_muscles).
        mean (ndarray): Mean vector used for centering during PCA.
        n_synergies (int): Number of components to use for reconstruction.

    Returns:
        reconstructed (ndarray): Reconstructed EMG data matrix, shape (n_samples, n_muscles).
    """

    print(f"\nReconstructing the signal with {n_synergies} synergies...")
    
    # Select the first n_components
    W_rec = W[:, :n_synergies]
    H_rec = H[:n_synergies, :]
    
    # Reconstruct the data
    reconstructed = np.dot(W_rec, H_rec) + mean

    print("Reconstruction completed.\n")

    return reconstructed



####################################################################################################
# NMF functions
####################################################################################################

#---------------------------------------------------------------------------------------------
def nmf_emg(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    """
    Applies Non-negative Matrix Factorization (NMF) to extract muscle synergies 
    and their activations from EMG data.

    Args:
        emg_data (ndarray): Input EMG data matrix of shape (n_muscle, n_samples).
        n_components (int): Number of synergies (components) to extract.
        init (str): Initialization method for NMF (e.g. for sparse, 'nndsvd', 'random').
        max_iter (int): Maximum number of iterations before stopping.
        l1_ratio (float): L1 regularization ratio (between 0 and 1).
        alpha_W (float): Regularization strength for the activation matrix U (sparsity coefficient).
        random_state (int): Random seed for reproducibility.

    Returns:
        W (ndarray): Muscle synergy matrix (muscle weights), shape (n_muscles, n_components).
        H (ndarray): Synergy activations over time (neural drive), shape (n_components, n_samples).
    """

    print("\nApplying NMF...")
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    H_T = nmf.fit_transform(emg_data)         # Synergy activations over time (Neural drive matrix)
    W_T = nmf.components_                     # Channels patterns (Muscular synergy activation matrix)
 
 
    # Transpose W and H to match the correct shapes if needed
    if H_T.shape[0] == emg_data.shape[0]:
        H = H_T.T         
    else:
        H = H_T
    if W_T.shape[0] == n_components:
        W = W_T.T 
    else:
        W = W_T   

    print("NMF completed.\n")

    return W, H



#---------------------------------------------------------------------------------------------
def nmf_emg_reconstruction(W, H, final_emg_for_nmf):
    """
    Reconstructs the EMG signal using a selected number of NMF components (synergies).
    
    Args:
        W (ndarray): Muscle synergy matrix (muscle weights), shape (n_muscles, n_components).
        H (ndarray): Synergy activations over time (neural drive), shape (n_components, n_samples).
        final_emg_for_nmf (ndarray): Original EMG data matrix used for NMF, shape (n_samples, n_muscles).
    Returns:
        reconstructed (ndarray): Reconstructed EMG data, shape (n_muscles, n_samples).
    """

    print(f"\nReconstructing the signal with selected number of synergies...")
    reconstructed_nmf = np.dot(W, H)
    # Ensure the reconstructed signal has the same shape as the original EMG data fro plotting comparisons
    if reconstructed_nmf.shape[0] != final_emg_for_nmf.shape[0]:
        reconstructed_nmf = reconstructed_nmf.T  # Transpose if needed to match (n_channels, n_samples)
    print("Reconstruction check completed.\n")

    return reconstructed_nmf



####################################################################################################
# General purpose synergy helper functions
####################################################################################################

#-------------------------------------------------------------------------------------------
# Function to scale the synergy activation matrix to the original EMG amplitude range (for plotting purposes)

def scale_synergy_signal(X, emg_data):
    """
    Normalize synergy activation matrix to the amplitude range of the original EMG.

    This ensures that synergy activations (X) can be compared or plotted in the 
    same scale as EMG signals.

    Args:
        X (ndarray): Activation matrix (n_samples x n_synergies).
        emg_data (ndarray): Original EMG signals (n_samples x n_channels).

    Returns:
        ndarray: Scaled activation matrix (same shape as X).
    """
    
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = ((X - X_min) / (X_max - X_min)) * (emg_max - emg_min) + emg_min
    X_scaled = np.maximum(X_scaled, 0)  # Ensures W_scaled is non-negative
    return X_scaled


#-------------------------------------------------------------------------------------------
# Functions to filter the data, has been developed 2 approaches: butterworth bandpass and notch

# Band-pass 20-500Hz, Notch 50Hz
# 1. Bandpass filter design
def butter_bandpass(signal, fs, lowcut=20, highcut=500, order=4):
    """Applies a Butterworth bandpass filter to the signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_b = filtfilt(b, a, signal)
    return filtered_b

# 2. Notch filter design
def notch_filter(signal, fs, n_freq=50.0, Q=30.0):
    """Applies a notch filter to remove powerline interference."""
    nyq = 0.5 * fs
    w0 = n_freq / nyq  
    b, a = iirnotch(w0, Q)
    filtered_n = filtfilt(b, a, signal)
    return filtered_n

# 3. RMS in 200 ms Windows 
def compute_rms(signal, window_size=200):
    """Computes the RMS of the signal using a moving window."""
    # RMS over sliding windows
    squared = np.power(signal, 2)
    window = np.ones(window_size)/window_size
    rms = np.sqrt(np.convolve(squared, window, mode='same'))
    return rms

# 4. Apply bandpass and notch filters to the signal + rms
def preprocess_emg(emg_signal, fs):
    bandpassed = butter_bandpass(emg_signal, fs)
    notch_removed = notch_filter(bandpassed, fs)
    rms_signal = compute_rms(notch_removed)
    return rms_signal


#-------------------------------------------------------------------------------------------
# Function to compute the Moore–Penrose pseudo-inverse of a matrix
def compute_pseudo_inverse(matrix):
    """
    Computes the Moore-Penrose pseudo-inverse of a matrix and plot insights.

    Args:
        matrix (ndarray): Input matrix of shape (n_samples, n_synergies) or similar.

    Returns:
        pseudo_inverse (ndarray): Pseudo-inverse of the input matrix.
    """
    print("\nComputing pseudo-inverse of the neural matrix W from specimen dataset...")
    pseudo_inverse = np.linalg.pinv(matrix)
    print("Input matrix shape:", matrix.shape)
    print("Pseudo-inverse shape:", pseudo_inverse.shape)
    print("Pseudo-inverse computation completed.\n")
    return pseudo_inverse



#-------------------------------------------------------------------------------------------

def find_max_difference(matrix):
    # Ensure the input is a numpy array for easier manipulation
    matrix = np.array(matrix)
    
    # Initialize variables to track the highest value and its index
    highest_value = float('-inf')
    highest_index = (-1, -1)  # To store the row and column index of the highest value
    
    # Iterate through each value in the matrix
    for i in range(matrix.shape[0]):  # Iterate over rows
        for j in range(matrix.shape[1]):  # Iterate over columns
            value = matrix[i, j]
            if value > highest_value:
                highest_value = value
                highest_index = (i, j)
    
    # Get the row and column of the highest value
    row, col = highest_index
    
    # Initialize corresponding value
    corresponding_value = None
    
    # Check if the next row exists
    if row + 1 < matrix.shape[0]:
        corresponding_value = matrix[row + 1, col]
    else:
        corresponding_value = matrix[row - 1, col]  # if the highest value is found in the last row, take the corresponding previous row's value
    
    # Calculate the maximum difference if corresponding value exists
    max_difference = None
    if row + 1 < matrix.shape[0]:
        max_difference = highest_value - corresponding_value
    else:
        max_difference = corresponding_value - highest_value
    
    return highest_value, corresponding_value, max_difference


#-------------------------------------------------------------------------------------------
def scale_differences(matrix, max_diff):
    """
    matrix: 2 x new_n_samples
    max_diff: scalar from find_max_difference
    """
    matrix = np.array(matrix)
    
    # Step 1: compute the differences (row 0 - row 1) for each column
    differences = matrix[0, :] - matrix[1, :]

    # Step 2: scale with respect to max_diff (avoid division by 0)
    if max_diff == 0:
        scaled = np.zeros_like(differences)  # All zeros if no difference possible
    else:
        scaled = differences / max_diff

    # Step 3: saturate values to stay between 0 and 1
    saturated = np.clip(scaled, 0, 1)

    # Step 4: reshape to 1 x new_n_samples
    return saturated.reshape(1, -1)


#-------------------------------------------------------------------------------------------
# Function to align baselines of multiple signals

def align_signal_baselines(signals, method='mean'):
    """
    Aligns all signals to the same baseline.

    Parameters:
    - signals: list of 1D numpy arrays
    - method: 'mean', 'first', or 'min'

    Returns:
    - list of aligned signals
    """
    if method == 'mean':
        offsets = [np.mean(s) for s in signals]
    elif method == 'first':
        offsets = [s[0] for s in signals]
    elif method == 'min':
        offsets = [np.min(s) for s in signals]
    else:
        raise ValueError("Invalid method")

    reference = np.mean(offsets)  # Align to the group mean
    return [s - (off - reference) for s, off in zip(signals, offsets)]

#-------------------------------------------------------------------------------------------
