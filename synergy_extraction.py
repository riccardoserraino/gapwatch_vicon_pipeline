from config import *

# Utils functions
from utils.utils_loading import *
from utils.utils_motion import * 
from utils.utils_synergies import *
from utils.utils_visual import *




########################################################################
# Initialization - Set ROS topic for Gapwatch and Vicon and set bagfile paths
########################################################################

# Topics
emg_topic = "/emg"  

# Paths
base_path = {
    "bag_emg": "dataset/bag_emg/"
}

train_dataset = "power_grasp1.bag"     # <-- Change here to use a different file
test_dataset = "power_grasp2.bag"      # <-- Change here to use a different file




########################################################################
# Data Loading - Read Gapwatch, Vicon data from selected ROS bag files
########################################################################

#-----------------------------------------------------------------------
# Load EMG train data from a test bag file 
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + train_dataset     

emg_data_train, timestamps_train = load_emg_data(bag_path_emg, emg_topic)

# Data Reshaping 1 - Reshape raw EMG vector into (16 x N) matrix format
final_emg_train, timestamps_emg_train, fs_emg_train = reshape_emg_data(emg_data_train, timestamps_train)
aligned_emg_train = np.array(align_signal_baselines(final_emg_train, method='mean'))
plot_emg(aligned_emg_train, title='Raw EMG Signals - Mean Value Aligned')


# Data Filtering 2 - Filter EMG data with EMG filtering specs
filtered_emg_train = np.array([preprocess_emg(final_emg_train[i, :], fs=fs_emg_train) for i in range(final_emg_train.shape[0])])
plot_emg(filtered_emg_train, title='Filtered EMG Signals')


#-----------------------------------------------------------------------
# Load EMG test data from a test bag file 
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + test_dataset     

emg_data_test, timestamps_test = load_emg_data(bag_path_emg, emg_topic)

# Data Reshaping 1 - Reshape raw EMG vector into (16 x N) matrix format
final_emg_test, timestamps_emg_test, fs_emg_test = reshape_emg_data(emg_data_test, timestamps_test)

# Data Filtering 2 - Filter EMG data with EMG filtering specs
filtered_emg_test = np.array([preprocess_emg(final_emg_test[i, :], fs=fs_emg_test) for i in range(final_emg_test.shape[0])])




########################################################################
# Gapwatch Data Processing - Extract synergies from the EMG train data 
#                          - Estimate the synergy activation patterns of the test data
#                          - Reconstruct the EMG test data from the extracted synergies
########################################################################

#------------------------------------------------------------------------
# Extract synergies from the EMG train data using Sparse NMF
#------------------------------------------------------------------------

optimal_synergies_nmf = 2

# Transpose for sklearn and plotting compatibility
final_emg_for_nmf = filtered_emg_train.T  

W, H = nmf_emg(final_emg_for_nmf, 
               n_components=optimal_synergies_nmf,
               init='nndsvd', 
               max_iter=500, 
               l1_ratio=0.15, 
               alpha_W=0.0005, 
               random_state=21)

# Plot original E, channel weights W, and activation over time H
reconstructed_nmf = nmf_emg_reconstruction(W, H, final_emg_for_nmf)
plot_nmf(final_emg_for_nmf, W, H, optimal_synergies_nmf)


# Print shapes of extracted matrices
print("Insights into extracted NMF matrices:")
print(f" - Final EMG for NMF shape: {final_emg_for_nmf.shape}")   # Should be (n_samples, n_channels)
print(f" - Extracted Synergy Matrix W shape: {W.shape}")          # Should be (n_channels, n_synergies)
print(f" - Extracted Activation Matrix H shape: {H.shape}\n")     # Should be (n_synergies, n_samples)


# Pseudo inverse of H matrix (neural matrix representing activation patterns)
W_pinv = compute_pseudo_inverse(W) # Should be (n_synergies, n_channels)

print("\nEstimating new synergy matrix H using pseudo-inverse of W and test...")
# Estimate the synergy matrix H from the pseudo-inverse of W
estimated_H = np.dot(W_pinv, filtered_emg_test)  
# Should be (n_synergies, testdata_n_samples) = (n_synergies, n_channels) X (n_channels, testdata_n_samples)
print("Estimation completed.\n")

# Print insights into the estimated synergy matrix
print("\nInsights into estimated synergy matrix:")
print(" - Pseudo-inverse of W shape:", W_pinv.shape)  # Should be (n_synergies, n_channels)
print(" - Filtered EMG test data shape:", filtered_emg_test.shape)  # Should be (n_channels, testdata_n_samples)
print(f" - Estimated Synergy Matrix H from W_pinv shape: {estimated_H.shape} \n")   # Should be (n_synergies, testdata_n_samples)


# Reconstruct the EMG test data using the estimated synergy matrix H
print("\nReconstructing the EMG test data using estimated synergy matrix H...")
H_train = H
H_test = estimated_H
reconstructed_t = nmf_emg_reconstruction(W, H_test, filtered_emg_test.T)
print(f" - Reconstructed EMG shape: {reconstructed_nmf.shape}\n") # Should be (testdata_n_samples, n_channels) after doing the transpose for plotting purposes
print("Reconstruction completed.\n")

plot_all_results(filtered_emg_test.T, reconstructed_t, W, H_test, optimal_synergies_nmf, title='Reconstruction of Original Test Data with Train Data Synergies')




########################################################################
# Sigma Matrix EMG - Compute the Sigma matrix for the EMG test data to define hand closure
########################################################################
# This approach has to be considered valid if the training dataset is the power_grasp1.bag
# Otherwise the approach needs changes to adapt to the new dataset to the train synegies weight patterns extracted

highest_value, correspondent_value, max_difference = find_max_difference(H_train)

sigma_emg = scale_differences(H_test, max_difference)
print("\nInsights into the flexion/extention synergy matrix:")
print(f" - Highest value in H_train row: {highest_value}")
print(f" - Corresponding value in H_train row+1: {correspondent_value}")
print(f" - Maximum difference in H_train: {max_difference}")
print(f" - Flexion/Extension synergy matrix shape: {sigma_emg.shape}\n")

plot_sigma_emg(sigma_emg, title='Sigma Matrix EMG')

