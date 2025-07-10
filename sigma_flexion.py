from config import *

# Utils functions
from utils.utils_loading import *
from utils.utils_motion import * 
from utils.utils_synergies import *
from utils.utils_visual import *
from utils.utils_loadcombined import *



########################################################################
# Initialization - Set ROS topic for Gapwatch and Vicon and set bagfile paths
########################################################################

# Topics
emg_topic = "/emg"  
vicon_topic_hand = "/tf"  
vicon_topic_marker = '/vicon/unlabeled_markers'


# Dictionary mapping numbers to dataset paths
dataset_options = {
    '1':   '/power_grasp1.bag',
    '2':   '/power_grasp2.bag',
    '3':   '/power_grasp3.bag',
    '4':   '/power_grasp4.bag',
    '5':   '/power_grasp5.bag',
    '6':   '/power_grasp6.bag',
    '7':   '/power_grasp7.bag',
    '8':   '/power_grasp8.bag',
    '9':   '/power_grasp9.bag',
    '10':  '/power_grasp10.bag',
    '11':  '/pinch1.bag',
    '12':  '/pinch2.bag',
    '13':  '/pinch3.bag',
    '14':  '/pinch4.bag',
    '15':  '/pinch5.bag',
    '16':  '/pinch6.bag',
    '17':  '/pinch7.bag',
    '18':  '/pinch8.bag',
    '19':  '/pinch9.bag',
    '20':  '/pinch10.bag',
    '21':  '/ulnar1.bag',
    '22':  '/ulnar2.bag',
    '23':  '/ulnar3.bag',
    '24':  '/ulnar4.bag',
    '25':  '/ulnar5.bag',
    '26':  '/ulnar6.bag',
    '27':  '/ulnar7.bag',
    '28':  '/ulnar8.bag',
    '29':  '/ulnar9.bag',
    '30':  '/ulnar10.bag',
    '31':  '/sto1.bag',
    '32':  '/sto2.bag',
    '33':  '/sto3.bag',
    '34':  '/sto4.bag',
    '35':  '/sto5.bag',
    '36':  '/sto6.bag',
    '37':  '/sto7.bag',
    '38':  '/sto8.bag',
    '39':  '/sto9.bag',
    '40':  '/sto10.bag',
    '41':  '/thumb_up1.bag',
    '42':  '/thumb_up2.bag',
    '43':  '/thumb_up3.bag',
    '44':  '/thumb_up4.bag',
    '45':  '/thumb_up5.bag',
    '46':  '/thumb_up6.bag',
    '47':  '/thumb_up7.bag',
    '48':  '/thumb_up8.bag',
    '49':  '/thumb_up9.bag',
    '50':  '/thumb_up10.bag',
    '51':  '/thumb1.bag',
    '52':  '/thumb2.bag',
    '53':  '/thumb3.bag',
    '54':  '/thumb4.bag',
    '55':  '/thumb5.bag',
    '56':  '/thumb6.bag',
    '57':  '/thumb7.bag',
    '58':  '/thumb8.bag',
    '59':  '/thumb9.bag',
    '60':  '/thumb10.bag',
    '61':  '/index1.bag',
    '62':  '/index2.bag',
    '63':  '/index3.bag',
    '64':  '/index4.bag',
    '65':  '/index5.bag',
    '66':  '/index6.bag',
    '67':  '/index7.bag',
    '68':  '/index8.bag',
    '69':  '/index9.bag',
    '70':  '/index10.bag',
    '71':  '/middle1.bag',
    '72':  '/middle2.bag',
    '73':  '/middle3.bag',
    '74':  '/middle4.bag',
    '75':  '/middle5.bag',
    '76':  '/middle6.bag',
    '77':  '/middle7.bag',
    '78':  '/middle8.bag',
    '79':  '/middle9.bag',
    '80':  '/middle10.bag',
    '81':  '/ring1.bag',
    '82':  '/ring2.bag',
    '83':  '/ring3.bag',
    '84':  '/ring4.bag',
    '85':  '/ring5.bag',
    '86':  '/ring6.bag',
    '87':  '/ring7.bag',
    '88':  '/ring8.bag',
    '89':  '/ring9.bag',
    '90':  '/ring10.bag',
    '91':  '/little1.bag',
    '92':  '/little2.bag',
    '93':  '/little3.bag',
    '94':  '/little4.bag',
    '95':  '/little5.bag',
    '96':  '/little6.bag',
    '97':  '/little7.bag',
    '98':  '/little8.bag',
    '99':  '/little9.bag',
    '100': '/little10.bag',
    '101': '/bottle1.bag',
    '102': '/bottle2.bag',
    '103': '/bottle3.bag',
    '104': '/bottle4.bag',
    '105': '/bottle5.bag',
    '106': '/bottle6.bag',
    '107': '/bottle7.bag',
    '108': '/bottle8.bag',
    '109': '/bottle9.bag',
    '110': '/bottle10.bag',
    '111': '/pen1.bag',
    '112': '/pen2.bag',
    '113': '/pen3.bag',
    '114': '/pen4.bag',
    '115': '/pen5.bag',
    '116': '/pen6.bag',
    '117': '/pen7.bag',
    '118': '/pen8.bag',
    '119': '/pen9.bag',
    '120': '/pen10.bag',
    '121': '/tablet1.bag',
    '122': '/tablet2.bag',
    '123': '/tablet3.bag',
    '124': '/tablet4.bag',
    '125': '/tablet5.bag',
    '126': '/tablet6.bag',
    '127': '/tablet7.bag',
    '128': '/tablet8.bag',
    '129': '/tablet9.bag',
    '130': '/tablet10.bag',
    '131': '/pinza1.bag',
    '132': '/pinza2.bag',
    '133': '/pinza3.bag',
    '134': '/pinza4.bag',
    '135': '/pinza5.bag',
    '136': '/pinza6.bag',
    '137': '/pinza7.bag',
    '138': '/pinza8.bag',
    '139': '/pinza9.bag',
    '140': '/pinza10.bag',
    '141': '/phone1.bag',
    '142': '/phone2.bag',
    '143': '/phone3.bag',
    '144': '/phone4.bag',
    '145': '/phone5.bag',
    '146': '/phone6.bag',
    '147': '/phone7.bag',
    '148': '/phone8.bag',
    '149': '/phone9.bag',
    '150': '/phone10.bag'
}






# Paths
base_path = {
    "bag_emg": "dataset/bag_emg",
    "bag_vicon": "dataset/bag_vicon",
}


# Ask user to select datasets and their order
selected_paths = select_datasets(dataset_options)


########################################################################
# Data Loading & Reshaping - Read Gapwatch, Vicon data from selected ROS bag files
########################################################################

#-----------------------------------------------------------------------
# Load EMG train data from a train bag file 
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + '/power_grasp1.bag'     

emg_data_train, timestamps_train = load_emg_data(bag_path_emg, emg_topic)

# Data Reshaping 1 - Reshape raw EMG vector into (16 x N) matrix format
final_emg_train, timestamps_emg_train, fs_emg_train = reshape_emg_data(emg_data_train, timestamps_train)

# Data Reshaping 2 - Filter EMG data
filtered_emg_train = np.array([preprocess_emg(final_emg_train[i, :], fs=fs_emg_train) for i in range(final_emg_train.shape[0])])


#-----------------------------------------------------------------------
# Load EMG, Vicon test data from a test bag file 
#-----------------------------------------------------------------------
pos_hand_final_combined = []
rot_hand_final_combined = []
pos_f1_final_combined =   []
pos_f2_final_combined =   []
pos_f3_final_combined =   []
pos_f4_final_combined =   []
pos_f5_final_combined =   []
final_emg_data_test_combined, final_timestamps_test_combined, fs_test_combined, pos_hand_final_combined, rot_hand_final_combined, pos_f1_final_combined, pos_f2_final_combined, pos_f3_final_combined, pos_f4_final_combined, pos_f5_final_combined = load_combined_reshape_vicon(selected_paths, emg_topic)
print("SHAPE: ", pos_hand_final_combined.shape)
print("SCIAMN: ", pos_hand_final_combined[:,1])


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
estimated_H = np.dot(W_pinv, final_emg_data_test_combined)  
# Should be (n_synergies, testdata_n_samples) = (n_synergies, n_channels) X (n_channels, testdata_n_samples)
print("Estimation completed.\n")

# Print insights into the estimated synergy matrix
print("\nInsights into estimated synergy matrix:")
print(" - Pseudo-inverse of W shape:", W_pinv.shape)  # Should be (n_synergies, n_channels)
print(" - Filtered EMG test data shape:", final_emg_data_test_combined.shape)  # Should be (n_channels, testdata_n_samples)
print(f" - Estimated Synergy Matrix H from W_pinv shape: {estimated_H.shape} \n")   # Should be (n_synergies, testdata_n_samples)


# Reconstruct the EMG test data using the estimated synergy matrix H
print("\nReconstructing the EMG test data using estimated synergy matrix H...")
H_train = H
H_test = estimated_H
reconstructed_t = nmf_emg_reconstruction(W, H_test, final_emg_data_test_combined.T)
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
for i in range(pos_hand_final_combined.shape[1]):

    # Loading of the current position of the hand and fingers wrt the world frame
    x1, y1, z1 = pos_hand_final_combined[:,i]
    x_f1, y_f1, z_f1 = pos_f1_final_combined[:,i]
    x_f2, y_f2, z_f2 = pos_f2_final_combined[:,i]
    x_f3, y_f3, z_f3 = pos_f3_final_combined[:,i]
    x_f4, y_f4, z_f4 = pos_f4_final_combined[:,i]
    x_f5, y_f5, z_f5 = pos_f5_final_combined[:,i]


    # Rotation matrix calculation to pass from the world frame to the hand frame
    rotation_matrix = from_q_to_rotation(rot_hand_final_combined[:,i])
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


#########################################################################
# Plotting Sigma matrices - comparison purpose
#########################################################################

# Insights into both sigma matrices
print("\nInsights into sigma motion matrix wrt to sigma synergy matrix")
print(f"Samples in Motion Sigma matrix: {len(sigma_motion)}")
print(f"Samples in EMG Sigma matrix: {final_timestamps_test_combined}\n")

# Ensure both signals are numpy arrays and translate sigma_motion for matching [0,1] range
sigma_emg = np.array(sigma_emg)

sigma_motion = np.array(sigma_motion)

# Compute the error between the two sigma matrices
sigma_error = np.abs(sigma_motion - sigma_emg)

# Comparison plot (transpose for plotting compatibility)
plot_sigma_matrices(sigma_motion.T, sigma_emg.T, sigma_error.T)


