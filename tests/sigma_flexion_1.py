# Hand model with no joints in the finger model and no inverse kinematics

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
vicon_topic_hand = "/tf"  
vicon_topic_marker = '/vicon/unlabeled_markers'

# Paths
base_path = {
    "bag_emg": "dataset/bag_emg/",
    "bag_vicon": "dataset/bag_vicon/",
    "npy_out": "dataset/npy_files/"
}

# Select the dataset you want to work on for calibrating synergies
train_dataset = "power_grasp1.bag"     # <-- Change here to use a different file
# Select the dataset you want to work on for testing 
test_dataset = "power_grasp2.bag"      # <-- Change here to use a different file




########################################################################
# Data Loading & Reshaping - Read Gapwatch, Vicon data from selected ROS bag files
########################################################################

#-----------------------------------------------------------------------
# Load EMG train data from a test bag file 
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + train_dataset     

emg_data_train, timestamps_train = load_emg_data(bag_path_emg, emg_topic)

# Data Reshaping 1 - Reshape raw EMG vector into (16 x N) matrix format
final_emg_train, timestamps_emg_train, fs_emg_train = reshape_emg_data(emg_data_train, timestamps_train)

# Data Reshaping 2 - Filter EMG data
filtered_emg_train = np.array([preprocess_emg(final_emg_train[i, :], fs=fs_emg_train) for i in range(final_emg_train.shape[0])])


#-----------------------------------------------------------------------
# Load EMG test data from a test bag file 
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + test_dataset     

emg_data_test, timestamps_test = load_emg_data(bag_path_emg, emg_topic)

# Data Reshaping 1 - Reshape raw EMG vector into (16 x N) matrix format
final_emg_test, timestamps_emg_test, fs_emg_test = reshape_emg_data(emg_data_test, timestamps_test)

# Data Reshaping 2 - Filter EMG data
filtered_emg_test = np.array([preprocess_emg(final_emg_test[i, :], fs=fs_emg_test) for i in range(final_emg_test.shape[0])])


#-----------------------------------------------------------------------
# Load Motion test data from a test bag file 
#-----------------------------------------------------------------------

bag_path_vicon = base_path['bag_vicon'] + test_dataset

# Extract hand and marker positions from the bag file
hand_positions, hand_orientations, marker_positions, timestamp_vicon = load_vicon_data(bag_path_vicon, vicon_topic_hand, vicon_topic_marker)

# Load the saved data from .bag files into lists
pos_hand = hand_positions  
rot_hand = hand_orientations
pos_f1 = marker_positions[0]
pos_f2 = marker_positions[1]
pos_f3 = marker_positions[2]
pos_f4 = marker_positions[3]
pos_f5 = marker_positions[4]

# Find the number of times you have to use the same vicon data to match the EMG data length
n_of_times = round(timestamps_emg_test/len(pos_hand))
print("Number of times to repeat Vicon data to match Gapwatch data:", n_of_times)




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
                alpha_W=0.005, 
                random_state=42)

# Uncomment the following lines for consistency check purposes
# To Reconstruct the EMG from extracted synergies
# Plot original E, reconstructed, channel weights W, and activation over time H
#reconstructed_nmf = nmf_emg_reconstruction(W, H, final_emg_for_nmf)
#plot_all_results(final_emg_for_nmf, reconstructed_nmf, W, H, optimal_synergies_nmf, title='NMF Synergy Extraction Results')


# Print shapes of extracted matrices
print("Insights into extracted NMF matrices:")
print(f" - Final EMG for NMF shape: {final_emg_for_nmf.shape}")   # Should be (n_samples, n_channels)
print(f" - Extracted Synergy Matrix W shape: {W.shape}")          # Should be (n_channels, n_synergies)
print(f" - Extracted Activation Matrix H shape: {H.shape}\n")     # Should be (n_synergies, n_samples)
#print(f" - Reconstructed EMG shape: {reconstructed_nmf.shape}\n") # Should be (n_samples, n_channels) after doing the transpose for plotting purposes



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
print("Reconstruction completed.\n")

#plot_all_results(filtered_emg_test.T, reconstructed_t, W, H_test, optimal_synergies_nmf)




########################################################################
# Sigma Matrix EMG - Compute the Sigma matrix for the EMG test data to define hand closure
########################################################################
# This approach has to be considered valid if the training dataset is the power_grasp1.bag
# Otherwise the approach needs changes to adapt to the new dataset to the train synegies weight patterns extracted

highest_value, correspondent_value, max_difference = find_max_difference(H_train)

sigma_emg = scale_differences(H_test, max_difference)
print("\nInsights into the flexion/extention synergy matrix O:")
print(f" - Highest value in H_train row: {highest_value}")
print(f" - Corresponding value in H_train row+1: {correspondent_value}")
print(f" - Maximum difference in H_train: {max_difference}")
print(f" - Flexion/Extension synergy matrix O shape: {sigma_emg.shape}\n")




########################################################################
# Vicon Data Processing - Extract hand and finger positions
#                       - Calculate angles of closure
#                       - Get sigma matrix values
########################################################################
# Hand model generation and animation

#-----------------------------------------------------------------------
# Initialization of the useful parameters for proccessing purposes
#----------------------------------------------------------------------- 
# Create the sigma matrix for vicon data
sigma_motion = []
max_angle_of_closure = 0
min_angle_of_closure = 160


#--------------------------------------------------------------------------
# Hand model initialization
#--------------------------------------------------------------------------
# The hand model is defined by a set of points that represent the structure of the hand and fingers.

# Definition of the hand points (positions relative to origin)
hand_points = {
    'h1': np.array([-0.025, 0.05, -0.01, 1]),
    'h2': np.array([-0.025, -0.035, -0.01, 1]),
    'h3': np.array([0.06, -0.035, -0.01, 1]),
    'h4': np.array([0.062, -0.01, -0.01, 1]),
    'h5': np.array([0.065, 0.02, -0.01, 1]),
    'h6': np.array([0.062, 0.05, -0.01, 1])
}

# Define the relative fingers position with respect to the hand frame
# These are the initial positions of the fingers in the hand frame
f1_rel_prev = np.array([0.15954817, -0.04850752, -0.01941226, 1])
f2_rel_prev = np.array([0.14718147, 0.12169835, -0.04258014, 1])
f3_rel_prev = np.array([0.02415149, 0.13794686, -0.09176365, 1])
f4_rel_prev = np.array([0.18140122, 0.06914709, -0.01804172, 1])
f5_rel_prev = np.array([1.82499501e-01, 9.96591903e-04, -1.92990169e-02, 1])
m_nulla = np.array([[1,0,0],[0,1,0],[0,0,1]])


#-----------------------------------------------------------------------
# Angles of closure - Sigma values computation
#-----------------------------------------------------------------------

for i in range(len(pos_hand)):
    
    # Loading of the current position of the hand and fingers wrt the world frame
    x1, y1, z1 = pos_hand[i]
    x_f1, y_f1, z_f1 = pos_f1[i]
    x_f2, y_f2, z_f2 = pos_f2[i]
    x_f3, y_f3, z_f3 = pos_f3[i]
    x_f4, y_f4, z_f4 = pos_f4[i]
    x_f5, y_f5, z_f5 = pos_f5[i]

    # Rotation matrix calculation
    # to pass from the world frame to the hand frame
    rotation_matrix = from_q_to_rotation(rot_hand[i])
    if(rotation_matrix == m_nulla).all():
        continue

    # Homogeneous transformation matrix
    # to make the base frame pass from the world frame to the hand frame
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = np.array([x1, y1, z1])

    # Calculate the relative position of the fingers in the hand frame
    f1 = np.linalg.inv(T) @ np.array([x_f1, y_f1, z_f1, 1])
    f2 = np.linalg.inv(T) @ np.array([x_f2, y_f2, z_f2, 1])
    f3 = np.linalg.inv(T) @ np.array([x_f3, y_f3, z_f3, 1])
    f4 = np.linalg.inv(T) @ np.array([x_f4, y_f4, z_f4, 1])
    f5 = np.linalg.inv(T) @ np.array([x_f5, y_f5, z_f5, 1])

    # Hungarian algorithm
    # To match the current finger positions with the previous ones
    P_prev = np.array([f1_rel_prev, f2_rel_prev, f3_rel_prev, f4_rel_prev, f5_rel_prev])
    P_new = np.array([f1, f2, f3, f4, f5])
    cost_matrix = cdist(P_prev, P_new) 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    f1_rel = P_new[col_ind[0]]
    f2_rel = P_new[col_ind[1]]
    f3_rel = P_new[col_ind[2]]
    f4_rel = P_new[col_ind[3]]
    f5_rel = P_new[col_ind[4]]

    # Update angle of closure of the fingers
    angles = [
        calculate_marker_angle(f1_rel, hand_points['h3'], [0.1, 0, -0.01]),  # Finger 1
        calculate_marker_angle(f5_rel, hand_points['h4'], [0.01, 0, -0.01]), # Finger 2
        calculate_marker_angle(f4_rel, hand_points['h5'], [0.01, 0, -0.01]), # Finger 3
        calculate_marker_angle(f2_rel, hand_points['h6'], [0.01, 0, -0.01]), # Finger 4
        calculate_marker_angle(f3_rel, hand_points['h1'], [0.01, 0, 0]) # Finger 5
    ]
    
    # Update sigma matrix value for the current frame
    sigma_value = normalize_angle(angles[0], angles[1], angles[2], angles[3], angles[4])
    for i in range(n_of_times):  # Every value is added n_of_times to the sigma list
        sigma_motion.append(sigma_value)

    # Save the position of the hand for the next iteration
    f1_rel_prev = f1_rel
    f2_rel_prev = f2_rel
    f3_rel_prev = f3_rel
    f4_rel_prev = f4_rel
    f5_rel_prev = f5_rel




#########################################################################
# Sigma matrix Motion - Compute the Sigma matrix for the Motion test data to define hand closure
#########################################################################

# Ensure the sigma list has the same number of elements as the EMG data
sigma_len = len(sigma_motion)
if sigma_len < timestamps_emg_test:
    for i in range(timestamps_emg_test - sigma_len):
        sigma_motion.append(sigma_value)

if sigma_len > timestamps_emg_test:
    sigma_motion = sigma_motion[:timestamps_emg_test]

sigma_len_final = len(sigma_motion)

print(len(sigma_motion), "Sigma Motion values generated")
print(timestamps_emg_test, "Samples in EMG data")




#########################################################################
# Plotting Sigma matrices - comparison purpose
#########################################################################

# Insights into both sigma matrices
print(f"Samples in Motion Sigma matrix: {sigma_len_final}")
print(f"Samples in EMG Sigma matrix: {timestamps_emg_test}")

# Ensure both signals are numpy arrays
sigma_emg = np.array(sigma_emg)
sigma_motion = np.array(sigma_motion)

# Compute the error between the two sigma matrices
sigma_error = np.abs(sigma_motion - sigma_emg)

# Comparison plot (transpose for plotting compatibility)
plot_sigma_matrices(sigma_motion.T, sigma_emg.T, sigma_error.T)




