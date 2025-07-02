# Import custom configuration and utility functions for EMG signal processing and plotting
from config import *
from synergy_extraction.utils_extraction import *
from synergy_extraction.utils_general import *
from synergy_extraction.utils_visual import *



########################################################################
# Initialization - Set ROS topic and available bagfile paths
########################################################################

selected_topic = '/emg'  # ROS topic to read EMG messages from

#-----------------------------------------------------------------------
# List of available bag files for EMG recordings
#-----------------------------------------------------------------------
# Calibration bag files
power_c = "dataset/power_grasp1.bag"
pinch_c = "dataset/pinch1.bag"
ulnar_c = "dataset/ulnar1.bag"
thumb_up_c = "dataset/thumb_up1.bag"
sto_c = "dataset/sto1.bag"

bottle_c = "dataset/bottle1.bag"
pen_c = "dataset/pen1.bag"
phone_c = "dataset/phone1.bag"
tablet_c = "dataset/tablet1.bag"
pinza_c = "dataset/pinza1.bag"

thumb_c = "dataset/thumb1.bag"
index_c = "dataset/index1.bag"
middle_c = "dataset/middle1.bag"
ring_c = "dataset/ring1.bag"
little_c = "dataset/little1.bag"

#-----------------------------------------------------------------------
# Test bag files (optional)
power = "dataset/power_grasp2.bag"
pinch = "dataset/pinch2.bag"
ulnar = "dataset/ulnar2.bag"
thumb_up = "dataset/thumb_up2.bag"
sto = "dataset/sto2.bag"

bottle = "dataset/bottle2.bag"
pen = "dataset/pen2.bag"
phone = "dataset/phone2.bag"
tablet = "dataset/tablet2.bag"
pinza = "dataset/pinza2.bag"

thumb = "dataset/thumb2.bag"
index = "dataset/index2.bag"
middle = "dataset/middle2.bag"
ring = "dataset/ring2.bag"
little = "dataset/little2.bag"



########################################################################
# Data loading - Read EMG data from selected ROS bag file
########################################################################

#-----------------------------------------------------------------------
# Load EMG calibration data from a specific bag file
#-----------------------------------------------------------------------

emg_data_calibration = []
timestamps_calibration = []

bag_path_calibration = power_c     # <-- Change calibration data here

print("\nLoading calibration data...")
# Open the bag and extract EMG values from messages
with rosbag.Bag(bag_path_calibration, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[selected_topic]):
        try:
            for i in msg.emg:  # Read each value in the EMG array
                emg_data_calibration.append(i)
                timestamps_calibration.append(t.to_sec())
        except AttributeError as e:
            print("Message missing expected fields:", e)
            break

emg_data_calibration = np.array(emg_data_calibration)
timestamps_calibration = np.array(timestamps_calibration)
print("Loading calibration data completed.\n")

#-----------------------------------------------------------------------
# Load EMG test data from a test bag file (optional)
#-----------------------------------------------------------------------

emg_data_test = []
timestamps_test = []

bag_path_test = power_c          # <-- Change test data here

print("\nLoading test data...")
with rosbag.Bag(bag_path_test, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[selected_topic]):
        try:
            for i in msg.emg:  # Read each value in the EMG array
                emg_data_test.append(i)
                timestamps_test.append(t.to_sec())
        except AttributeError as e:
            print("Message missing expected fields:", e)
            break

emg_data_test = np.array(emg_data_test)
timestamps_test = np.array(timestamps_test)
print("Loading test data completed.\n")



########################################################################
# Calibration Data Processing & Filtering - Reshape raw EMG vector into 
#                                           (16 x N) matrix format
########################################################################
# The bag file streams data as a flat list

#-----------------------------------------------------------------------
# Calibration data reshaping
#-----------------------------------------------------------------------

selector = 0
raw_emg_c = np.empty((16, 0))  # Initialize empty matrix with 16 rows (channels)

# Loop over all complete sets of 16-channel samples
print("\nReshaping calibration data into 16-channel matrix...")
for i in range(int(len(emg_data_calibration)/16)):
    temp = emg_data_calibration[selector:selector+16]   # Extract 16 consecutive samples
    new_column = np.array(temp).reshape(16, 1)          # Convert to column format
    raw_emg_c = np.hstack((raw_emg_c, new_column))      # Append column to EMG matrix
    selector += 16                                      # Move to next block
    print("Sample number: ", i)
print("Reshaping Calibration data completed.\n")


# Print shape information of calibration data
print("Insights into calibration data:")
print(f" - Acquired EMG Calibration data shape: {emg_data_calibration.shape}")  # Should be (n_samples * 16, )
print(f" - Calibration Reshaped EMG shape: {raw_emg_c.shape }")                 # Should be (n_samples, n_channels)

reshaped_timestamps_c = timestamps_calibration[::16]                  
reshaped_timestamps_int_c = len(reshaped_timestamps_c)        

print(f" - Calibration Timestamps count: {reshaped_timestamps_int_c}")   # Should be (n_samples)
duration_c = reshaped_timestamps_c[-1] - reshaped_timestamps_c[0]
print(f" - Duration of Calibration data: {duration_c:.2f} s")


#-----------------------------------------------------------------------
# Calibration data filtering
#-----------------------------------------------------------------------
# Calibration data filtering
c_fs=reshaped_timestamps_int_c/duration_c
print(f" - Sampling frequency fs of Calibration data : {c_fs:.2f} Hz\n")

# Band-pass + Notch filtering + rms
filtered_emg_c = np.array([preprocess_emg(raw_emg_c[i, :], fs=c_fs) for i in range(raw_emg_c.shape[0])])



########################################################################
# Test Data Processing & Filtering - Reshape raw EMG vector into 
#                                    (16 x N) matrix format
########################################################################
# The bag file streams data as a flat list

#-----------------------------------------------------------------------
# Test data reshaping
#-----------------------------------------------------------------------

selector_ = 0
raw_emg_t = np.empty((16, 0))  # Initialize empty matrix with 16 rows (channels)

# Loop over all complete sets of 16-channel samples
print("\nReshaping test data into 16-channel matrix...")
for i in range(int(len(emg_data_test)/16)):
    temp_ = emg_data_test[selector_:selector_+16]        # Extract 16 consecutive samples
    new_column_ = np.array(temp_).reshape(16, 1)         # Convert to column format
    raw_emg_t = np.hstack((raw_emg_t, new_column_))      # Append column to EMG matrix
    selector_ += 16                                      # Move to next block
    #print("Sample number: ", i)
print("Reshaping test data completed.\n")


# Print shape information of calibration data
print("Insights into test data:")
print(f" - Acquired EMG Test data shape: {emg_data_test.shape}")  # Should be (n_samples * 16, )
print(f" - Test Reshaped EMG shape: {raw_emg_t.shape }")          # Should be (n_samples, n_channels)

reshaped_timestamps_t = timestamps_test[::16]                  
reshaped_timestamps_int_t = len(reshaped_timestamps_t)        

print(f" - Test Timestamps count: {reshaped_timestamps_int_t}")   # Should be (n_samples)
duration_t = reshaped_timestamps_t[-1] - reshaped_timestamps_t[0]
print(f" - Duration of Test data: {duration_t:.2f} s")


#-----------------------------------------------------------------------
# Test data filtering
#-----------------------------------------------------------------------
# Calibration data filtering
c_fs=reshaped_timestamps_int_t/duration_t
print(f" - Sampling frequency fs of Test data : {c_fs:.2f} Hz\n")

# Band-pass + Notch filtering + rms
filtered_emg_t = np.array([preprocess_emg(raw_emg_t[i, :], fs=c_fs) for i in range(raw_emg_t.shape[0])])



########################################################################
# Data Plotting - Plot first insights into EMG data aquired from ROS bag (optional)
########################################################################

# Plot all raw channels in a single plot
#plot_all_channels(raw_emg, title='Raw EMG Channels')         
#plot_emg_channels_2cols(raw_emg)

# Filtering insights section--------------------------------------------
# Plotting with filter applied to raw data
#plot_raw_vs_filtered_channels_2cols(raw_emg, filtered_emg, title='Raw vs Band-pass & Notch Filtered EMG Channels')
#plot_all_channels(filtered_emg, title='Band-pass & Notch Filtered EMG Channels')
#plot_emg_channels_2cols(filtered_emg)



########################################################################
# PCA Synergy Extraction (optional, may need little fixes)
########################################################################

# Apply Principal Component Analysis (PCA) to extract synergies from EMG (filtered data, no alignment introduced)
#optimal_synergies_pca = 2
#max_components_pca = 16
#final_emg_for_pca = filtered_emg.T  # Transpose for sklearn compatibility (samples as rows)

# Decompose EMG into synergy components and reconstruct signal
#H, W, mean, rec = pca_emg(final_emg_for_pca, optimal_synergies_pca, random_state=42, svd_solver='full')
#reconstructed_pca = pca_emg_reconstruction(W, H, mean, optimal_synergies_pca)

# Plot original, reconstructed, and synergy data
#plot_all_results(final_emg_for_pca, reconstructed_pca, W, H, optimal_synergies_pca)





########################################################################
# Sparse NMF Synergy Extraction
########################################################################

# Apply Sparse Non-negative Matrix Factorization (NMF) to extract synergies from EMG Calibration (filtered data, no alignment introduced)
optimal_synergies_nmf = 2
max_synergies_nmf = 16

# Transpose for sklearn and plotting compatibility
final_emg_for_nmf = filtered_emg_c.T  

W, H = nmf_emg(final_emg_for_nmf, n_components=optimal_synergies_nmf,
                 init='nndsvd', max_iter=500, l1_ratio=0.15, alpha_W=0.005, random_state=42)

# Reconstruct the EMG from extracted synergies
# For consistency check purposes
reconstructed_nmf = nmf_emg_reconstruction(W, H, final_emg_for_nmf)

# Plot original, reconstructed, and synergy data
#plot_all_results(final_emg_for_nmf, reconstructed_nmf, W, H, optimal_synergies_nmf, title='NMF Synergy Extraction Results')


# Print shapes of extracted matrices
print("Insights into extracted NMF matrices:")
print(f" - Final EMG for NMF shape: {final_emg_for_nmf.shape}")   # Should be (n_samples, n_channels)
print(f" - Extracted Synergy Matrix W shape: {W.shape}")          # Should be (n_channels, n_synergies)
print(f" - Extracted Activation Matrix H shape: {H.shape}\n")     # Should be (n_synergies, n_samples)
print(f" - Reconstructed EMG shape: {reconstructed_nmf.shape}\n") # Should be (n_samples, n_channels) after doing the transpose for plotting purposes



########################################################################
# Pseudo-inverse of W matrix and estimation of new synergy matrix H
########################################################################

# Pseudo inverse of H matrix (neural matrix representing activation patterns)
W_pinv = compute_pseudo_inverse(W) # Should be (n_synergies, n_channels)

print("\nEstimating new synergy matrix H using pseudo-inverse of calibration W and test dataset...")
# Estimate the synergy matrix H from the pseudo-inverse of W
estimated_H = np.dot(W_pinv, filtered_emg_t)  # Should be (n_synergies, testdata_n_samples) = (n_synergies, n_channels) X (n_chaannels, testdata_n_samples)
print("Estimation completed.\n")

# Print insights into the estimated synergy matrix
print("Insights into estimated synergy matrix:")
print(" - Pseudo-inverse of W shape:", W_pinv.shape)  # Should be (n_synergies, n_channels)
print(" - Filtered EMG test data shape:", filtered_emg_t.shape)  # Should be (n_channels, testdata_n_samples)
print(f" - Estimated Synergy Matrix H from W_pinv shape: {estimated_H.shape} \n")   # Should be (n_synergies, testdata_n_samples)


########################################################################
# Finding the flexion/extention synergy matrix O
########################################################################
# Matrix O should be a mtrix of shape (1, testdata_n_sample) with values between 0 and 1 
# representing the flexion/extention pattern over time 



H_calibration = H
H_test = estimated_H
reconstructed_t = nmf_emg_reconstruction(W, H_test, filtered_emg_t.T)

plot_all_results(filtered_emg_t.T, reconstructed_t, W, H_test, optimal_synergies_nmf)

highest_value, correspondent_value, max_difference = find_max_difference(H_calibration)

O = scale_differences(H_test, max_difference)
print("\nInsights into the flexion/extention synergy matrix O:")
print(f" - Highest value in calibration H: {highest_value}")
print(f" - Corresponding value in test H: {correspondent_value}")
print(f" - Maximum difference: {max_difference}")
print(f" - Flexion/Extension synergy matrix O shape: {O.shape}\n")

plot_sigma_matrix(O, title='Flexion/Extension Synergy Matrix O')


'''
H_calibration = H
H_test = estimated_H

# Plotting the synergy activation patterns over time of the Calibration data
channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # Assuming 16 EMG channels, adjust as needed
for i in range(2):
    plt.plot(W[:, i], 'o-', label=f'Synergy {i+1}')
plt.title('Synergy Weighting Patterns')
plt.xlabel('EMG Channel')
plt.ylabel('Weight')
plt.legend(loc='upper right', ncol=2)
plt.xticks(channels) 


# To define which synegy dominates throughout the Estimated synegy matrix H_test
dominant_synergy = np.argmax(H_test, axis=0)
dominant_matrix = np.zeros_like(H_test)
dominant_matrix[dominant_synergy, np.arange(H_test.shape[1])] = 1
plot_dominant_synergy_line(dominant_synergy, title='Dominant Synergy Line') 

# To define which synergy can be interpreted as flexor
diff_calib = H_calibration[0, :] - H_calibration[1, :]
mean_diff = np.mean(diff_calib)
# if mean_diff > 0:
#     flexor_synergy = 0
print("mean diff calibration W", mean_diff)




'''
