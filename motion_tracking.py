from config import *

# Utils functions
from utils.utils_loading import *
from utils.utils_motion import * 
from utils.utils_synergies import *
from utils.utils_visual import *


########################################################################
# Initialization - Set ROS topic for Gapwatch and Vicon and set bagfile paths
########################################################################

emg_topic = "/emg"  
vicon_topic_hand = "/tf"  
vicon_topic_marker = '/vicon/unlabeled_markers'

base_path = {
    "bag_emg": "dataset/bag_emg/",
    "bag_vicon": "dataset/bag_vicon/",
    "npy_out": "dataset/npy_files/"
}

# Select the dataset you want to work on for testing 
test_dataset = "power_grasp2.bag"      # <-- Change here to use a different file



########################################################################
# Data loading - Read Gapwatch, Vicon data from selected ROS bag file
########################################################################

#-----------------------------------------------------------------------
# Load EMG test data from a test bag file (optional)
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + test_dataset     

emg_data_specimen, timestamps_specimen = load_emg_data(bag_path_emg, emg_topic)

# Data Processing - Reshape raw EMG vector into (16 x N) matrix format
raw_emg, timestamps_emg, fs_emg = reshape_emg_data(emg_data_specimen, timestamps_specimen)


#-----------------------------------------------------------------------
# Load EMG test data from a test bag file (optional)
#-----------------------------------------------------------------------

# Load Vicon data from a specific bag file
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
n_of_times = round(timestamps_emg/len(pos_hand))
print("\nSince Vicon fs is much lower than Gapwatch fs we need to reshape the sigma_motion values.")
print(" - Number of times to repeat Vicon data foo sigma_motion matrix: ", n_of_times)


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

plt.ion()

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.125, 0.25)
ax.set_ylim(-0.10, 0.175)
ax.set_zlim(-0.20, 0.05)


#--------------------------------------------------------------------------
# Hand model initialization
#--------------------------------------------------------------------------
# The hand model is defined by a set of points and lines that represent the structure of the hand and fingers.

# Definition of the visual elements of the hand
# name = ax.plot([x],[y],[z], 'color/shape', size)
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

min_length = min(len(pos_hand), len(rot_hand), len(pos_f1), len(pos_f2), len(pos_f3), len(pos_f4), len(pos_f5))


# Definition of reference hand points
pos_point(point_h1, -0.01, 0.01, -0.025)
pos_point(point_h2, -0.01, -0.06, -0.025)
pos_point(point_h3, 0.083, -0.05, -0.025)
pos_point(point_h4, 0.09, -0.01, -0.025)
pos_point(point_h5, 0.09, 0.03, -0.025)
pos_point(point_h6, 0.07, 0.07, -0.025)
pos_line(line_h1, point_h1, point_h2)
pos_line(line_h2, point_h2, point_h3)
pos_line(line_h3, point_h3, point_h4)
pos_line(line_h4, point_h4, point_h5)
pos_line(line_h5, point_h5, point_h6)
pos_line(line_h6, point_h6, point_h1)


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


# Initialize chain artists, used to clear from the previous iteration chains representation
# 'chain artists': The connected series of graphical elements that make up each finger's visual representation
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

    # Remove the previous representation of the fingers
    if finger_line_little is not None:
        finger_line_little.remove()
        finger_scatter_little.remove()
        finger_line_ring.remove()
        finger_scatter_ring.remove()
        finger_line_middle.remove()
        finger_scatter_middle.remove()
        finger_line_index.remove()
        finger_scatter_index.remove()
        finger_line_thumb.remove()
        finger_scatter_thumb.remove()
    

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
    

    # Plot new chain
    finger_line_little, = ax.plot([node[0] for node in nodes_little], [node[1] for node in nodes_little], [node[2] for node in nodes_little], color='black', linewidth=3)
    finger_scatter_little = ax.scatter([node[0] for node in nodes_little], [node[1] for node in nodes_little], [node[2] for node in nodes_little], color='black', s=5)
    finger_line_ring, = ax.plot([node[0] for node in nodes_ring], [node[1] for node in nodes_ring], [node[2] for node in nodes_ring], color='black', linewidth=3)
    finger_scatter_ring = ax.scatter([node[0] for node in nodes_ring], [node[1] for node in nodes_ring], [node[2] for node in nodes_ring], color='black', s=5)
    finger_line_middle, = ax.plot([node[0] for node in nodes_middle], [node[1] for node in nodes_middle], [node[2] for node in nodes_middle], color='black', linewidth=3)
    finger_scatter_middle = ax.scatter([node[0] for node in nodes_middle], [node[1] for node in nodes_middle], [node[2] for node in nodes_middle], color='black', s=5)
    finger_line_index, = ax.plot([node[0] for node in nodes_index], [node[1] for node in nodes_index], [node[2] for node in nodes_index], color='black', linewidth=3)
    finger_scatter_index = ax.scatter([node[0] for node in nodes_index], [node[1] for node in nodes_index], [node[2] for node in nodes_index], color='black', s=5)
    finger_line_thumb, = ax.plot([node[0] for node in nodes_thumb], [node[1] for node in nodes_thumb], [node[2] for node in nodes_thumb], color='black', linewidth=3)
    finger_scatter_thumb = ax.scatter([node[0] for node in nodes_thumb], [node[1] for node in nodes_thumb], [node[2] for node in nodes_thumb], color='black', s=5)


    # Get angle of closures 
    x_h1, y_h1, z_h1 = point_h1.get_data_3d()
    x_h2, y_h2, z_h2 = point_h2.get_data_3d()
    x_h3, y_h3, z_h3 = point_h3.get_data_3d()
    x_h4, y_h4, z_h4= point_h4.get_data_3d()
    x_h5, y_h5, z_h5 = point_h5.get_data_3d()
    x_h6, y_h6, z_h6, = point_h6.get_data_3d()

    angle_little = get_closure_angle_plot(point_h3, nodes_little, np.array([x_h3[0] + 0.1, y_h3[0], z_h3[0]]))
    angle_ring = get_closure_angle_plot(point_h4, nodes_ring, np.array([0.16272325, 0.00817894, z_h4[0]]))
    angle_middle = get_closure_angle_plot(point_h5, nodes_middle, np.array([0.17139323, 0.06217872, z_h5[0]]))
    angle_index = get_closure_angle_plot(point_h6, nodes_index, np.array([0.12973476, 0.09361607, z_h6[0]]))
    angle_thumb = get_closure_angle_plot(point_h1, nodes_thumb, np.array([-0.03948546, 0.08916224, z_h1[0]]))


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


    fig.canvas.draw()
    fig.canvas.flush_events()

    # For animation purposes the fingers are kept at the first position for letting the 
    # virtual hand start the movement from a still position first, instead of starting to plot the
    # animation only after the first vicon sample
    if i == 0:
        time.sleep(0.01)
    else: 
        time.sleep(0.0001)


plt.ioff()
plt.show()




########################################################################
# Sigma matrix processing - Little modification to sigma samples length before plotting 
########################################################################

# Ensure the sigma list has the same number of elements as the EMG data
sigma_len = len(sigma_motion)
if sigma_len < timestamps_emg:
    for i in range(timestamps_emg - sigma_len):
        sigma_motion.append(sigma_value)

if sigma_len > timestamps_emg:
    sigma_motion = sigma_motion[:timestamps_emg]

# Reaassign after modifications
sigma_len = len(sigma_motion)


print(f" - Sigma Motion samples: {sigma_len}")
print(f" - EMG data samples: {timestamps_emg}\n")

plot_sigma_motion(sigma_motion)


