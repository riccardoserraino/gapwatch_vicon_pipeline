from config import *

###########################################################################################################
# General purpose functions
###########################################################################################################


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

# Band-pass 10-500Hz, Notch 50Hz
# 1. Bandpass filter design
def butter_bandpass(signal, fs, lowcut=20, highcut=500, order=5):
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
    Computes the Moore-Penrose pseudo-inverse of a matrix.

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

def compute_flexion_extension_signals(H, max_flex=None, max_ext=None):
    H = np.array(H)
    flex = H[0, :]
    ext = H[1, :]

    if max_flex is None:
        max_flex = np.max(flex)
    if max_ext is None:
        max_ext = np.max(ext)

    norm_flex = np.clip(flex / max_flex, 0, 1)
    norm_ext = np.clip(ext / max_ext, 0, 1)

    return norm_flex.reshape(1, -1), norm_ext.reshape(1, -1)


def compute_sigma_ref(H, max_diff=None):
    diff = H[0, :] - H[1, :]
    if max_diff is None:
        max_diff = np.max(np.abs(diff))
    sigma = diff / max_diff  # Now range approx [-1, 1]
    sigma = np.clip((sigma + 1) / 2, 0, 1)  # Rescale to [0,1]
    return sigma.reshape(1, -1)


#-------------------------------------------------------------------------------------------


