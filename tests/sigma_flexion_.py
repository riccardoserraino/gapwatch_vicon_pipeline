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

#-----------------------------------------------------------------------
# Initialization of the useful parameters for proccessing purposes
#----------------------------------------------------------------------- 
sigma_motion = []
max_angle_of_closure = 0
min_angle_of_closure = 160


#--------------------------------------------------------------------------
# Hand model initialization
#--------------------------------------------------------------------------
# The hand model is defined by a set of points that represent the structure of the hand and fingers.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.2)

# Definition of the visual elements of the hand
#name = ax.plot([x],[y],[z], 'color/shape', size)
origin,   = ax.plot([], [], [], 'bo', markersize=5)
point_h1, = ax.plot([], [], [], 'bo', markersize=5)
point_h2, = ax.plot([], [], [], 'bo', markersize=5)
point_h3, = ax.plot([], [], [], 'bo', markersize=5)
point_h4, = ax.plot([], [], [], 'bo', markersize=5)
point_h5, = ax.plot([], [], [], 'bo', markersize=5)
point_h6, = ax.plot([], [], [], 'bo', markersize=5)
line_h1, = ax.plot([], [], [], 'k-', linewidth=2)
line_h2, = ax.plot([], [], [], 'k-', linewidth=2)
line_h3, = ax.plot([], [], [], 'k-', linewidth=2)
line_h4, = ax.plot([], [], [], 'k-', linewidth=2)
line_h5, = ax.plot([], [], [], 'k-', linewidth=2)
line_h6, = ax.plot([], [], [], 'k-', linewidth=2)

point_h1 = [-0.01, 0.01, -0.025]
point_h2 = [-0.01, -0.06, -0.025]
point_h3 = [0.083, -0.05, -0.025]
point_h4 = [0.09, -0.01, -0.025]
point_h5 = [0.09, 0.03, -0.025]
point_h6 = [0.07, 0.07, -0.025]


# Definition of fingers length
little_length = 0.085
ring_length = 0.1
middle_length = 0.13
index_length =  0.1
thumb_length = 0.14

# Finger chains ddefinition for inverse kinematic model
# Little definition
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

# Ring definition
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

# Midlle definition
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

# Index definition
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

# Thumb definition
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


# Define the relative fingers position with respect to the hand frame
# These are the initial positions of the fingers in the hand frame
f1_rel_prev = np.array([0.15954817, -0.04850752, -0.01941226, 1])
f2_rel_prev = np.array([0.14718147, 0.12169835, -0.04258014, 1])
f3_rel_prev = np.array([0.02415149, 0.13794686, -0.09176365, 1])
f4_rel_prev = np.array([0.18140122, 0.06914709, -0.01804172, 1])
f5_rel_prev = np.array([1.82499501e-01, 9.96591903e-04, -1.92990169e-02, 1])
m_nulla = np.array([[1,0,0],[0,1,0],[0,0,1]])

# Initialize chain artists, used to clear from the previous iteration chains representation
finger_line_little = None
finger_scatter_little = None
finger_line_ring = None
finger_scatter_ring = None
finger_line_middle = None
finger_scatter_middle = None
finger_line_index = None
finger_scatter_index = None
finger_line_thumb = None
finger_scatter_thumb = None


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
    
    
    # Compute inverse kinematics
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
    
    
    # Get angle of closures 
    x_h1, y_h1, z_h1 = point_h1.get_data_3d()
    x_h2, y_h2, z_h2 = point_h2.get_data_3d()
    x_h3, y_h3, z_h3 = point_h3.get_data_3d()
    x_h4, y_h4, z_h4= point_h4.get_data_3d()
    x_h5, y_h5, z_h5 = point_h5.get_data_3d()
    x_h6, y_h6, z_h6, = point_h6.get_data_3d()


    angle_little = get_closure_angle_plot(point_h3, nodes_little, np.array([x_h3[0] + 0.1, y_h3[0], z_h3[0]]))
    angle_ring = get_closure_angle_plot(point_h4, nodes_ring, np.array([0.16272325, 0.00817894, z_h4[0]]))
    angle_middle = get_closure_angle(point_h5, nodes_middle, np.array([0.17139323, 0.06217872, z_h5[0]]))
    angle_index = get_closure_angle(point_h6, nodes_index, np.array([0.12973476, 0.09361607, z_h6[0]]))
    angle_thumb = get_closure_angle(point_h1, nodes_thumb, np.array([-0.03948546, 0.08916224, z_h1[0]]))

    
    # Update sigma matrix value for the current frame
    sigma_value = normalize_angle(angle_little, angle_ring, angle_middle, angle_index, angle_thumb)
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




