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
test_dataset = "pen9.bag"      # <-- Change here to use a different file




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


min_length = min(len(pos_hand), len(rot_hand), len(pos_f1), len(pos_f2), len(pos_f3), len(pos_f4), len(pos_f5))




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




########################################################################
# Vicon Data Processing - Extract Hand and Finger Positions
#                       - Plot Animation
#                       - Calculate Angles of Closure
#                       - Compute the Sigma matrix for the Motion test data to define hand closure
########################################################################

#-----------------------------------------------------------------------
# Initialization of the useful parameters for plotting and analysis
#----------------------------------------------------------------------- 
# Create the sigma matrix for vicon data
sigma_motion = []
max_angle_of_closure = 0
min_angle_of_closure = 160


#--------------------------------------------------------------------------
# Hand model initialization
#--------------------------------------------------------------------------

point_h1 = [-0.01, 0.01, -0.025]
point_h2 = [-0.01, -0.06, -0.025]
point_h3 = [0.083, -0.05, -0.025]
point_h4 = [0.09, -0.01, -0.025]
point_h5 = [0.09, 0.03, -0.025]
point_h6 = [0.07, 0.07, -0.025]


# Definition of initial pose for consistency in plotting (defined wrt to the subject's hand neutral position)
f1_rel_prev = np.array([0.13868391, -0.05139445,  0.00857016, 1])
f2_rel_prev = np.array([0.11976245,  0.13734637, -0.00763299, 1])
f3_rel_prev = np.array([-0.0369033, 0.15533912, -0.02849389, 1])
f4_rel_prev = np.array([0.16185893, 0.08565602, 0.008876, 1])
f5_rel_prev = np.array([0.16463813, 0.03239017, 0.01738579, 1])
m_nulla = np.array([[1,0,0],[0,1,0],[0,0,1]])


#--------------------------------------------------------------------------
# Inverse Kinematics useful definition
#--------------------------------------------------------------------------

# Definition of fingers total length
little_length = 0.085
ring_length = 0.13
middle_length = 0.14
index_length =  0.12
thumb_length = 0.17



# Little finger definition chain
little_chain = Chain(name='ring', links=[
    URDFLink(
        name="joint1",
        translation_vector=[0.083, -0.05, -0.025],
        orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(np.pi/3, np.pi)
    ),
    URDFLink(
        name="joint2",
        translation_vector=[0, 0, little_length*0.5],
        orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint3",
        translation_vector=[0, 0, little_length*0.34],
        orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint4",
        translation_vector=[0, 0, little_length*0.26],
        orientation=[0, 0, 0],
        rotation=[0, 1, 0]
    )
])

# Ring finger definition chain
ring_chain = Chain(name='ring', links=[
    URDFLink(
        name="joint1",
        translation_vector=[0.09, -0.01, -0.025],
        orientation=[0, 0, 0],
        rotation=[-0.2425, 0.9701, 0],
        bounds=(np.pi/3, np.pi)
    ),
    URDFLink(
        name="joint2",
        translation_vector=[0, 0, ring_length*0.5],
        orientation=[0, 0, 0],
        rotation=[-0.2425, 0.9701, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint3",
        translation_vector=[0, 0, ring_length*0.34],
        orientation=[0, 0, 0],
        rotation=[-0.2425, 0.9701, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint4",
        translation_vector=[0, 0, ring_length*0.26],
        orientation=[0, 0, 0],
        rotation=[-0.2425, 0.9701, 0]
    )
])

# Midlle finger definition chain
middle_chain = Chain(name='middle', links=[
    URDFLink(
        name="joint1",
        translation_vector=[0.09, 0.03, -0.025],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0],
        bounds=(np.pi/3, np.pi)
    ),
    URDFLink(
        name="joint2",
        translation_vector=[0, 0, middle_length*0.5],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint3",
        translation_vector=[0, 0, middle_length*0.34],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint4",
        translation_vector=[0, 0, middle_length*0.26],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0]
    )
])

# Index finger definition chain
index_chain = Chain(name='index', links=[
    URDFLink(
        name="joint1",
        translation_vector=[0.07, 0.07, -0.025],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0],
        bounds=(np.pi/3, np.pi)
    ),
    URDFLink(
        name="joint2",
        translation_vector=[0, 0, index_length*0.5],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint3",
        translation_vector=[0, 0, index_length*0.34],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint4",
        translation_vector=[0, 0, index_length*0.26],
        orientation=[0, 0, 0],
        rotation=[-0.36765889, 0.92996072, 0]
    )
])

# Thumb finger definition chain 
thumb_chain = Chain(name='thumb', links=[
    URDFLink(
        name="joint1",
        translation_vector=[-0.01, 0.01, -0.025],
        orientation=[0, 0, 0],
        rotation=[-0.68558295, -0.68558295, -0.24485105],
        bounds=(np.pi/3, np.pi)
    ),
    URDFLink(
        name="joint2",
        translation_vector=[0, 0, thumb_length*0.34],
        orientation=[0, 0, 0],
        rotation=[-0.68558295, -0.68558295, -0.24485105],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint3",
        translation_vector=[0, 0, thumb_length*0.34],
        orientation=[0, 0, 0],
        rotation=[-0.68558295, -0.68558295, -0.24485105],
        bounds=(0, np.pi/2)
    ),
    URDFLink(
        name="joint4",
        translation_vector=[0, 0, thumb_length*0.26],
        orientation=[0, 0, 0],
        rotation=[-0.68558295, -0.68558295, -0.24485105]
    )
])


#--------------------------------------------------------------------------
# Animation loop
#--------------------------------------------------------------------------
for i in range(min_length):

    # Loading of the current position of the hand and fingers wrt the world frame
    x1, y1, z1 = pos_hand[i]
    x_f1, y_f1, z_f1 = pos_f1[i]
    x_f2, y_f2, z_f2 = pos_f2[i]
    x_f3, y_f3, z_f3 = pos_f3[i]
    x_f4, y_f4, z_f4 = pos_f4[i]
    x_f5, y_f5, z_f5 = pos_f5[i]


    # Rotation matrix calculation to pass from the world frame to the hand frame
    rotation_matrix = from_q_to_rotation(rot_hand[i])
    if(rotation_matrix == m_nulla).all():
        continue
    
    # Homogeneous transformation matrix to pass from the world frame to the hand frame
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = np.array([x1, y1, z1])

    # Calculate the relative position of the fingers in the hand frame
    f1 = np.linalg.inv(T) @ np.array([x_f1, y_f1, z_f1, 1])
    f2 = np.linalg.inv(T) @ np.array([x_f2, y_f2, z_f2, 1])
    f3 = np.linalg.inv(T) @ np.array([x_f3, y_f3, z_f3, 1])
    f4 = np.linalg.inv(T) @ np.array([x_f4, y_f4, z_f4, 1])
    f5 = np.linalg.inv(T) @ np.array([x_f5, y_f5, z_f5, 1])


    # Hungarian algorithm to match the current finger positions with the previous ones
    # This is done to avoid the problem of the fingers jumping from one position to another
    P_prev = np.array([f1_rel_prev, f2_rel_prev, f3_rel_prev, f4_rel_prev, f5_rel_prev])
    P_new = np.array([f1, f2, f3, f4, f5])
    cost_matrix = cdist(P_prev, P_new) 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    f1_rel = P_new[col_ind[0]]
    f2_rel = P_new[col_ind[1]]
    f3_rel = P_new[col_ind[2]]
    f4_rel = P_new[col_ind[3]]
    f5_rel = P_new[col_ind[4]]


    # Compute inverse kinematics for fingers
    little_position = [f1_rel[0], f1_rel[1], f1_rel[2]]
    ring_position = [f5_rel[0], f5_rel[1], f5_rel[2]]
    middle_position = [f4_rel[0], f4_rel[1], f4_rel[2]]
    index_position = [f2_rel[0], f2_rel[1], f2_rel[2]]
    thumb_position = [f3_rel[0], f3_rel[1], f3_rel[2]]
    little_solution = little_chain.inverse_kinematics(little_position)
    ring_solution = ring_chain.inverse_kinematics(ring_position)
    middle_solution = ring_chain.inverse_kinematics(middle_position)
    index_solution = index_chain.inverse_kinematics(index_position)
    thumb_solution = thumb_chain.inverse_kinematics(thumb_position)


    # Compute forward kinematics for all links to obtain the joint positions
    transformation_matrixes_little = little_chain.forward_kinematics(little_solution, full_kinematics=True)
    nodes_little = [tm[:3, 3] for tm in transformation_matrixes_little]  # Extract translation vectors
    transformation_matrixes_ring = ring_chain.forward_kinematics(ring_solution, full_kinematics=True)
    nodes_ring = [tm[:3, 3] for tm in transformation_matrixes_ring]  # Extract translation vectors
    transformation_matrixes_middle = middle_chain.forward_kinematics(middle_solution, full_kinematics=True)
    nodes_middle = [tm[:3, 3] for tm in transformation_matrixes_middle]  # Extract translation vectors
    transformation_matrixes_index = index_chain.forward_kinematics(index_solution, full_kinematics=True)
    nodes_index = [tm[:3, 3] for tm in transformation_matrixes_index]  # Extract translation vectors
    transformation_matrixes_thumb = thumb_chain.forward_kinematics(thumb_solution, full_kinematics=True)
    nodes_thumb = [tm[:3, 3] for tm in transformation_matrixes_thumb]  # Extract translation vectors
    
    
    # Compute angle of closure
    angle_little = get_closure_angle(point_h3, nodes_little, np.array([point_h3[0] + 0.1, point_h3[1], point_h3[2]]))
    angle_ring = get_closure_angle(point_h4, nodes_ring, np.array([0.16272325, 0.00817894, point_h4[2]]))
    angle_middle = get_closure_angle(point_h5, nodes_middle, np.array([0.17139323, 0.06217872, point_h5[2]]))
    angle_index = get_closure_angle(point_h6, nodes_index, np.array([0.12973476, 0.09361607, point_h6[2]]))
    angle_thumb = get_closure_angle(point_h1, nodes_thumb, np.array([-0.03948546, 0.08916224, point_h1[2]]))


    # Here all the angles are normalized to express the sigma value of closure of the hand
    sigma_value = normalize_angle(angle_little, angle_ring, angle_middle, angle_index, angle_thumb)
    # Every value is added n_of_times to the sigma list, to match the EMG data length
    for i in range(n_of_times):                 
        sigma_motion.append(sigma_value)


    # Update the previous position of the fingers
    f1_rel_prev = f1_rel
    f2_rel_prev = f2_rel
    f3_rel_prev = f3_rel
    f4_rel_prev = f4_rel
    f5_rel_prev = f5_rel




########################################################################
# Sigma matrix Motion - Compute the Sigma matrix for the Motion test data to define hand closure
########################################################################

# Ensure the sigma list has the same number of elements as the EMG data
sigma_len = len(sigma_motion)
if sigma_len < timestamps_emg_test:
    for i in range(timestamps_emg_test - sigma_len):
        sigma_motion.append(sigma_value)

if sigma_len > timestamps_emg_test:
    sigma_motion = sigma_motion[:timestamps_emg_test]

# Reaassign after modifications
sigma_mot_len = len(sigma_motion)


print(f" - Sigma Motion samples: {sigma_len}")
print(f" - EMG data samples: {timestamps_emg_test}\n")




#########################################################################
# Plotting Sigma matrices - comparison purpose
#########################################################################

# Insights into both sigma matrices
print("\nInsights into sigma motion matrix wrt to sigma synergy matrix")
print(f"Samples in Motion Sigma matrix: {sigma_mot_len}")
print(f"Samples in EMG Sigma matrix: {timestamps_emg_test}\n")

# Ensure both signals are numpy arrays and translate sigma_motion for matching [0,1] range
sigma_emg = np.array(sigma_emg)

sigma_motion = np.array(sigma_motion)

# Compute the error between the two sigma matrices
sigma_error = np.abs(sigma_motion - sigma_emg)

# Comparison plot (transpose for plotting compatibility)
plot_sigma_matrices(sigma_motion.T, sigma_emg.T, sigma_error.T)


