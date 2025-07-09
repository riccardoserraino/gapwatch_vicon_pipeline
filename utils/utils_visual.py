from config import *

###########################################################################################################
# Visualize helper functions
###########################################################################################################

#-------------------------------------------------------------------------------------------
def scale_synergy_signal(X, emg_data):
    """
    Normalize synergy activation matrix to the amplitude range of the original EMG.

    This ensures that synergy activations (X) can be compared or plotted in the 
    same scale as EMG signals.

    Args:
        X (ndarray): Activation matrix (n_samples x n_synergies).
        emg_data (ndarray): Original EMG signals (n_samples x n_channels).

    Returns:
        X_scaled: Scaled activation matrix (same shape as X).
    """
    
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = ((X - X_min) / (X_max - X_min)) * (emg_max - emg_min) + emg_min
    X_scaled = np.maximum(X_scaled, 0)  # Ensures W_scaled is non-negative
    return X_scaled


#-------------------------------------------------------------------------------------------
def plot_emg(emg_data, title=''):
    """
    Plot all EMG channels on a single graph for a global overview.

    Args:
        emg_data (ndarray): 2D array of EMG data with shape (n_channels, n_samples).
    """

    plt.figure(figsize=(8, 3))
 
    for j in range(emg_data.shape[0]):
        x = np.linspace(0, emg_data.shape[1] , emg_data.shape[1])
        plt.plot(x, emg_data[j], label='Channel {}'.format(j))
    #plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (mV)")
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_emg_channels_2cols(emg_data, title=''):
    """
    Plot each EMG channel in a separate subplot, organized into two columns.

    Args:
        emg_data (ndarray): 2D array of EMG data with shape (n_emg_channels, n_samples).
    """

    n_channels, n_samples = emg_data.shape

    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(8, 8), sharex=True)
    time = np.linspace(0, n_samples, n_samples)

    for i in range(16):
        row = i % 8
        col = i // 8
        ax = axes[row, col]
        ax.plot(time, emg_data[i])
        ax.set_title(f'Channel {i}')
    
        if row == 7:
            ax.set_ylabel("Amplitude (mV)")
            ax.set_xlabel("Time (samples)")
    #fig.suptitle(title)
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_raw_vs_filtered_channels_2cols(raw_emg, filtered_emg, title=''):
    """
    Plot raw and filtered EMG signals in subplots, organized into two columns.

    Args:
        raw_emg (ndarray): Raw EMG data, shape (n_channels, n_samples)
        filtered_emg (ndarray): Filtered EMG data, same shape
    """

    n_channels, n_samples = raw_emg.shape
    time = np.linspace(0, n_samples, n_samples)

    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(6, 8), sharex=True)

    for i in range(n_channels):
        row = i % 8
        col = i // 8
        ax = axes[row, col]

        ax.plot(time, raw_emg[i], label='Raw', alpha=0.6)
        ax.plot(time, filtered_emg[i], label='Filtered')
        ax.set_title(f'Channel {i}', fontsize=8)

        if row == 7:
            ax.set_xlabel("Time (samples)")
        if col == 0:
            ax.set_ylabel("Amplitude (mV)")

        ax.legend(fontsize=6)
    #fig.suptitle(title)

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_all_results(emg_data, E_reconstructed, W, H, selected_synergies, title=''):
    """
    Plot a comprehensive overview of EMG signal decomposition using synergies.

    This function generates four stacked subplots:
    1. Original EMG signals.
    2. Reconstructed EMG signals from synergies.
    3. Time-varying activation of each synergy.
    4. Synergy-to-muscle weight distributions.

    Args:
        emg_data (ndarray): Raw EMG data (n_samples x n_muscles).
        E_reconstructed (ndarray): Reconstructed EMG from synergy model (same shape).
        W (ndarray): Synergy activation matrix (n_samples x n_synergies).
        H (ndarray): Synergy weights matrix (n_synergies x n_emg_channel).
        selected_synergies (int): Number of synergies used in the model.
    """
    
    W_scaled = scale_synergy_signal(W, emg_data)

    channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # Assuming 16 EMG channels, adjust as needed

    plt.figure(figsize=(8, 8))

    # Panel 1: Original EMG Signals
    plt.subplot(4, 1, 1)
    plt.plot(emg_data)
    plt.title(f'Original Test Data')
    plt.ylabel('Amplitude (mV)')
    plt.xlabel('Time (samples)')
    plt.xticks()

    
    # Panel 2: Reconstructed EMG Signals
    plt.subplot(4, 1, 2)
    plt.plot(E_reconstructed)
    plt.title(f'Reconstructed EMG ({selected_synergies} Synergies)')
    plt.ylabel('Amplitude (mV)')
    plt.xlabel('Time (samples)')
    plt.xticks()

    
    # Panel 3: Synergy Activation Patterns over time
    plt.subplot(4, 1, 3)
    for i in range(selected_synergies):
        plt.plot(W_scaled[:, i], 'o-', label=f'Synergy {i+1}')
    plt.title('Synergy Weighting Patterns')
    plt.xlabel('EMG Channel')
    plt.ylabel('Weight')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks(channels)  

    
    # Panel 4: Synergy Weighting Patterns
    plt.subplot(4, 1, 4)
    for i in range(selected_synergies):
        plt.plot(H[i, :],  label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Amplitude (mV)')
    plt.xlabel('Time (samples)')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks()
    
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_nmf(emg_data,  W, H, selected_synergies, title=''):
    """
    Plot a comprehensive overview of EMG signal decomposition using synergies.

    This function generates four stacked subplots:
    1. Original EMG signals.
    2. Time-varying activation of each synergy.
    3. Synergy-to-muscle weight distributions.

    Args:
        emg_data (ndarray): Raw EMG data (n_samples x n_muscles).
        W (ndarray): Synergy activation matrix (n_samples x n_synergies).
        H (ndarray): Synergy weights matrix (n_synergies x n_emg_channel).
        selected_synergies (int): Number of synergies used in the model.
    """
    W_scaled = scale_synergy_signal(W, emg_data)

    channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # Assuming 16 EMG channels, adjust as needed

    plt.figure(figsize=(8, 6))

    # Panel 1: Original EMG Signals
    plt.subplot(3, 1, 1)
    plt.plot(emg_data)
    plt.title('EMG Filtered Signal (E)')
    plt.ylabel('Amplitude (mV)')
    plt.xlabel('Time (samples)')
    plt.xticks()

    
    # Panel 2: Synergy Activation Patterns over time
    plt.subplot(3, 1, 2)
    for i in range(selected_synergies):
        plt.plot(W[:, i], 'o-', label=f'Synergy {i+1}')
    plt.title('Synergy Weighting Patterns (S)')
    plt.xlabel('EMG Channel')
    plt.ylabel('Weight')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks(channels)  

    
    # Panel 3: Synergy Weighting Patterns
    plt.subplot(3, 1, 3)
    for i in range(selected_synergies):
        plt.plot(H[i, :],  label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time (U)')
    plt.ylabel('Amplitude (mV)')
    plt.xlabel('Time (samples)')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks()
    
    plt.tight_layout()
    plt.show()



#-------------------------------------------------------------------------------------------
def plot_sigma_matrices(sigma_motion, sigma_emg, sigma_error):
    """
    Plots sigma matrices comparison.

    Parameters: 
        - sigma_motion: matrix defining hand closure obtained from vicon data.
        - sigma_emg: matrix defining hand closure obtained from gapwatch data.
        - sigma_error: matrix defining error between the two input matrices.
    """
    
    plt.figure(figsize=(8, 6))

    # Plot Sigma EMG
    plt.subplot(3, 1, 1)
    plt.plot(sigma_motion, label='Sigma Motion (Vicon)', color='b')
    plt.ylabel('Sigma EMG')
    plt.ylim(0, 1.1)
    plt.title('Synergy-based Hand Closure Estimation (GapWatch)')
    plt.legend()

    # Plot Sigma Motion
    plt.subplot(3, 1, 2)
    plt.plot(sigma_emg, label='Sigma EMG (GapWatch)', color='g')
    plt.ylabel('Sigma Motion')
    plt.ylim(0, 1.1)
    plt.title('Motion-based Hand Closure Estimation (Vicon)')
    plt.legend()

    # Plot Sigma Error
    plt.subplot(3, 1, 3)
    plt.plot(sigma_error, label='Error |Sigma Motion - Sigma EMG|', color='r')
    plt.xlabel('Time (samples)')
    plt.ylabel('Absolute Error')
    plt.ylim(0, 1.1)
    plt.title('Difference Between EMG and Motion Sigma Matrices')
    plt.legend()

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------

def plot_sigma_emg(scaled_matrix, title="Flexion-Extention Matrix From EMG Analysis"):
    """
    Plots a 1 x N matrix showing values between 0 and 1.
    
    Parameters:
    - scaled_matrix: 1 x N numpy array
    """

    scaled_matrix = np.array(scaled_matrix)
    
    if scaled_matrix.shape[0] != 1:
        raise ValueError("Input must be a 1-row matrix (1 x new_n_samples)")

    values = scaled_matrix.flatten()
    x = np.arange(len(values))

    plt.figure(figsize=(8, 2.5))
    plt.plot(values, color='g')
    plt.ylim(0, 1.1)  # leave some space above 1
    plt.xlabel("Time (samples)")
    plt.xticks()
    plt.ylabel("Flexion-Extention Value")
    #plt.title(title)

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_sigma_motion(sigma_motion, title='Flexion-Extention Matrix from Motion Analysis'):
    """
    Plots a 1 x N matrix showing values between 0 and 1.

    Parameters:
    - scaled_matrix: 1 x N numpy array
    """


    plt.figure(figsize=(8, 2.5))
    plt.plot(sigma_motion, color='b')
    plt.ylim(0, 1.1)  # leave some space above 1
    plt.xlabel("Time (samples)")
    plt.ylabel("Flexion-Extention Value")
    plt.xticks()
    #plt.title(title)

    plt.tight_layout()
    plt.show()




#-------------------------------------------------------------------------------------------
def plot_signals(original, bandpassed, notch_removed, rms_signal, fs, channel_number):
    time = np.linspace(0, len(original) / fs, len(original))
    
    plt.figure(figsize=(8, 6))

    plt.subplot(4, 1, 1)
    plt.plot(time, original, label=f'Channel {channel_number}')
    plt.title('Original EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time, bandpassed, label=f'Channel {channel_number}')
    plt.title('Bandpassed EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time, notch_removed, label=f'Channel {channel_number}')
    plt.title('Notch Filtered EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time, rms_signal, label=f'Channel {channel_number}')
    plt.title('RMS of Notch Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()






