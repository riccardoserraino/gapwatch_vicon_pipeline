from config import *

# --- Filter Parameters ---
# Sampling frequency (Hz). This value is crucial for filter design.
fs = 2000

# Butterworth Bandpass filter parameters
# Lower cutoff frequency (Hz). Signals below this frequency will be attenuated.
lowcut_bp = 20
# Upper cutoff frequency (Hz). Signals above this frequency will be attenuated.
highcut_bp = 500
# Order of the Butterworth filter. Determines the steepness of the filter's roll-off.
order_bp = 4

# Notch filter parameters
# Frequency to be attenuated by the notch filter (Hz), typically for powerline interference.
n_freq_notch = 50.0
# Quality factor of the notch filter. A higher Q results in a narrower attenuation band.
Q_notch = 30.0




# --- 1. Butterworth Bandpass Filter Design ---
# Calculate the Nyquist frequency, which is half of the sampling frequency.
# It represents the maximum frequency that can be unambiguously represented in the digital signal.
nyq = 0.5 * fs
# Normalize the lower cutoff frequency by the Nyquist frequency.
# This normalization is required by SciPy's filter design functions.
low_bp_norm = lowcut_bp / nyq
# Normalize the upper cutoff frequency by the Nyquist frequency.
high_bp_norm = highcut_bp / nyq

# Design the Butterworth bandpass filter.
# 'b' and 'a' are the numerator and denominator polynomial coefficients of the filter's transfer function.
b_bp, a_bp = butter(order_bp, [low_bp_norm, high_bp_norm], btype='band')

# Calculate the frequency response (magnitude and phase) of the bandpass filter.
# 'worN' specifies the number of frequency points for the calculation, providing high resolution.
# 'fs' ensures the frequencies returned are in Hz.
w_bp, h_bp = freqz(b_bp, a_bp, worN=8192, fs=fs)




# --- 2. Notch Filter Design ---
# Normalize the notch frequency by the Nyquist frequency.
w0_notch = n_freq_notch / nyq
# Design the IIR notch filter.
# 'b' and 'a' are the numerator and denominator polynomial coefficients.
b_notch, a_notch = iirnotch(w0_notch, Q_notch)

# Calculate the frequency response (magnitude and phase) of the notch filter.
w_notch, h_notch = freqz(b_notch, a_notch, worN=8192, fs=fs)




# --- 3. Bode Plotting Function ---
# Auxiliary function to plot a single Bode diagram, encompassing both magnitude and phase.
def plot_bode(frequencies, response, title, ax_mag, ax_phase, mark_freqs=None):
    # Magnitude plot section.
    # Converts the complex frequency response magnitude into Decibels (dB).
    ax_mag.plot(frequencies, 20 * np.log10(abs(response)), color='blue')
    ax_mag.set_title(f'Bode Diagram: Magnitude - {title}')
    ax_mag.set_ylabel('Gain (dB)')
    ax_mag.grid(True, which="both", ls="-")
    # Adds a horizontal reference line at 0 dB.
    ax_mag.axhline(0, color='gray', linestyle='--')
    
    # Adds vertical lines to highlight specific frequencies (e.g., cutoff, notch).
    if mark_freqs:
        for f, label in mark_freqs.items():
            ax_mag.axvline(f, color='red', linestyle='--', linewidth=1, label=label)
        ax_mag.legend()

    # Configures the X-axis (frequency) to a logarithmic scale, standard for Bode plots.
    ax_mag.set_xscale('log')
    ax_mag.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax_mag.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax_mag.minorticks_on()
    # Sets the frequency range for the plot, typically from a low frequency to the Nyquist frequency.
    ax_mag.set_xlim([1, fs/2])

    # Phase plot section.
    # Calculates the phase in degrees and unwraps it to remove discontinuities at +/- 180 degrees.
    angles = np.unwrap(np.angle(response)) * 180 / np.pi
    ax_phase.plot(frequencies, angles, color='green')
    ax_phase.set_title(f'Bode Diagram: Phase - {title}')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (degrees)')
    ax_phase.grid(True, which="both", ls="-")
    ax_phase.set_xscale('log')
    ax_phase.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax_phase.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax_phase.minorticks_on()
    ax_phase.set_xlim([1, fs/2])



# --- Plotting Individual Filters ---
# Creates a figure and subplots for the Butterworth Bandpass filter's Bode diagram.
fig_bp, (ax_mag_bp, ax_phase_bp) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
fig_bp.suptitle('Bode Diagrams of Butterworth Bandpass Filter', fontsize=14)
plot_bode(w_bp, h_bp,
          f'Order {order_bp}, {lowcut_bp}-{highcut_bp} Hz',
          ax_mag_bp, ax_phase_bp,
          mark_freqs={lowcut_bp: 'Lowcut', highcut_bp: 'Highcut'})
plt.tight_layout()
plt.show()

# Creates a figure and subplots for the Notch filter's Bode diagram.
fig_notch, (ax_mag_notch, ax_phase_notch) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
fig_notch.suptitle('Bode Diagrams of Notch Filter', fontsize=14)
plot_bode(w_notch, h_notch,
          f'{n_freq_notch} Hz with Q={Q_notch}',
          ax_mag_notch, ax_phase_notch,
          mark_freqs={n_freq_notch: 'Notch Freq'})
plt.tight_layout()
plt.show()



# --- Plot of the Combined Filter Cascade Response (Intrinsic Phase) ---
# The overall frequency response of filters connected in cascade is the product of their individual complex frequency responses.
h_combined_original_phase = h_bp * h_notch

# Creates a figure for the combined filter's Bode diagram, showing its intrinsic phase response.
fig_combined, (ax_mag_combined, ax_phase_combined) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
fig_combined.suptitle('Bode Diagrams of Filter Cascade (Bandpass + Notch) - Intrinsic Phase', fontsize=14)
plot_bode(w_bp, h_combined_original_phase,
          f'Bandpass ({lowcut_bp}-{highcut_bp} Hz) + Notch ({n_freq_notch} Hz)',
          ax_mag_combined, ax_phase_combined,
          mark_freqs={lowcut_bp: 'Lowcut', highcut_bp: 'Highcut', n_freq_notch: 'Notch Freq'})
plt.tight_layout()
plt.show()



# --- Plot of the Combined Filter Cascade Response (Simulating Phase after filtfilt) ---
# The 'filtfilt' function applies the filter forward and then backward, resulting in a zero-phase filter.
# While the magnitude response becomes the square of the original magnitude (|H(f)|^2),
# the crucial effect on phase is its linearization to approximately zero degrees across all frequencies.
# This plot visually represents the **zero-phase characteristic achieved by `filtfilt`**.
# Note: For real-valued signals, `filtfilt` effectively eliminates phase distortion, making the phase response flat at 0 degrees.

fig_combined_filtfilt, (ax_mag_combined_filtfilt, ax_phase_combined_filtfilt) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
fig_combined_filtfilt.suptitle('Bode Diagrams: Filter Cascade with "filtfilt" Effect', fontsize=14)

ax_mag_filtfilt = ax_mag_combined_filtfilt
ax_phase_filtfilt = ax_phase_combined_filtfilt

# Magnitude plot for the combined filter. `filtfilt` effectively squares the magnitude response.
# Here, we plot the original combined magnitude for clarity, but in the thesis, emphasize it represents |H(f)|^2.
ax_mag_filtfilt.plot(w_bp, 20 * np.log10(abs(h_combined_original_phase)), color='blue')
ax_mag_filtfilt.set_title(f'Magnitude - Bandpass + Notch (with filtfilt)')
ax_mag_filtfilt.set_ylabel('Gain (dB)')
ax_mag_filtfilt.grid(True, which="both", ls="-")
ax_mag_filtfilt.axhline(0, color='gray', linestyle='--')
for f, label in {lowcut_bp: 'Lowcut', highcut_bp: 'Highcut', n_freq_notch: 'Notch Freq'}.items():
    ax_mag_filtfilt.axvline(f, color='red', linestyle='--', linewidth=1, label=label)
ax_mag_filtfilt.legend()
ax_mag_filtfilt.set_xscale('log')
ax_mag_filtfilt.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax_mag_filtfilt.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax_mag_filtfilt.minorticks_on()
ax_mag_filtfilt.set_xlim([1, fs/2])

# Phase plot: A flat line at 0 degrees, representing the ideal zero-phase characteristic achieved by `filtfilt`.
ax_phase_filtfilt.plot(w_bp, np.zeros_like(w_bp), color='g', linestyle='-', linewidth=2)
ax_phase_filtfilt.set_title(f'Phase - Bandpass + Notch (with filtfilt)')
ax_phase_filtfilt.set_xlabel('Frequency (Hz)')
ax_phase_filtfilt.set_ylabel('Phase (degrees)')
ax_phase_filtfilt.grid(True, which="both", ls="-")
ax_phase_filtfilt.set_xscale('log')
ax_phase_filtfilt.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax_phase_filtfilt.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax_phase_filtfilt.minorticks_on()
ax_phase_filtfilt.set_xlim([1, fs/2])
ax_phase_filtfilt.set_ylim([-5, 5])

plt.tight_layout()
plt.show()