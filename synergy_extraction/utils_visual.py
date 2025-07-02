from config import *
from synergy_extraction.utils_general import *

###########################################################################################################
# Visualize helper functions
###########################################################################################################


#-------------------------------------------------------------------------------------------
def plot_all_channels(emg_data, title=''):
    """
    Plot all EMG channels on a single graph for a global overview.

    Args:
        emg_data (ndarray): 2D array of EMG data with shape (n_channels, n_samples).

    Returns:
        None. Displays a matplotlib figure.
    """

    plt.figure(figsize=(8, 6)) 
    for j in range(emg_data.shape[0]):
        x = np.linspace(0, emg_data.shape[1] , emg_data.shape[1])
        plt.plot(x, emg_data[j], label='Channel {}'.format(j))
    plt.title("EMG signal overview")
    plt.xlabel("Samples over time")
    plt.ylabel("Channel activation")
    if title: 
        plt.title(title)
    plt.legend(loc='best', fontsize='small', markerscale=1)
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_emg_channels_2cols(emg_data, ):
    """
    Plot each EMG channel in a separate subplot, organized into two columns.

    Args:
        emg_data (ndarray): 2D array of EMG data with shape (n_emg_channels, n_samples).

    Returns:
        None. Displays a matplotlib figure with subplots for each channel.
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
            ax.set_ylabel("Activation")
            ax.set_xlabel("Time (samples)")
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_raw_vs_filtered_channels_2cols(raw_emg, filtered_emg, title=''):
    """
    Plot raw and filtered EMG signals in subplots, organized into two columns.

    Args:
        raw_emg (ndarray): Raw EMG data, shape (n_channels, n_samples)
        filtered_emg (ndarray): Filtered EMG data, same shape

    Returns:
        None. Displays matplotlib figure.
    """

    n_channels, n_samples = raw_emg.shape
    time = np.linspace(0, n_samples, n_samples)

    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(10, 10), sharex=True)

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
            ax.set_ylabel("Activation")

        ax.legend(fontsize=6)
    fig.suptitle(title)

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

    Returns:
        None. Displays a matplotlib figure with 4 subplots.
    """
    
    print(f'\nPlotting results...\n\n')

    #W_scaled = scale_synergy_signal(W, emg_data)

    channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # Assuming 16 EMG channels, adjust as needed

    plt.figure(figsize=(10, 8))
    plt.suptitle(title, fontsize=14)

    # Panel 1: Original EMG Signals
    plt.subplot(4, 1, 1)
    plt.plot(emg_data)
    plt.title('Original EMG Signals')
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
        plt.plot(W[:, i], 'o-', label=f'Synergy {i+1}')
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


def plot_sigma_matrix(scaled_matrix, title="Flexion-Extention Matrix"):
    """
    Plots a 1 x N matrix showing values between 0 and 1.
    
    Parameters:
    - scaled_matrix: 1 x N numpy array
    - title: optional title for the plot
    """

    scaled_matrix = np.array(scaled_matrix)
    
    if scaled_matrix.shape[0] != 1:
        raise ValueError("Input must be a 1-row matrix (1 x new_n_samples)")

    values = scaled_matrix.flatten()
    x = np.arange(len(values))

    plt.figure(figsize=(10, 4))
    plt.plot(x, values, color='g')
    plt.ylim(0, 1.1)  # leave some space above 1
    plt.xlabel("Time (samples)")
    plt.ylabel("Flexion-Extention Value")
    plt.title(title)

    plt.tight_layout()
    plt.show()



#-------------------------------------------------------------------------------------------

def plot_dominant_synergy_line(dominant_synergy, title="Dominant Synergy Over Time"):
    x = np.arange(len(dominant_synergy))
    y = dominant_synergy

    plt.figure(figsize=(10, 3))
    plt.plot(x, y, drawstyle='steps-mid', color='purple', linewidth=2)
    plt.yticks([0, 1], ['Synergy 0', 'Synergy 1'])
    plt.xlabel("Sample Index")
    plt.ylabel("Dominant Synergy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------

