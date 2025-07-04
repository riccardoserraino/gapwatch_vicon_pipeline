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
print("Number of times to repeat vicon data:", n_of_times)



########################################################################
# Vicon Data Processing - Extract Hand and Finger pPositions
#                       - Calculate Angles of Closure
#                       - Compute the Sigma matrix for the Motion test data to define hand closure
########################################################################

#-----------------------------------------------------------------------
# Initialization of the useful parameters for plotting and analysis
#----------------------------------------------------------------------- 
# Create the sigma matrix for vicon data
sigma_vicon = []
max_angle_of_closure = 0
min_angle_of_closure = 160

plt.ion()

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.2)


#--------------------------------------------------------------------------
# Hand model initializations
#--------------------------------------------------------------------------
# The hand model is defined by a set of points and lines that represent the structure of the hand and fingers.

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

# Definition of the geometrical structure of the hand
pos_point(origin, 0, 0, 0)
pos_point(point_h1, -0.025, 0.05, -0.01)
pos_point(point_h2, -0.025, -0.035, -0.01)
pos_point(point_h3, 0.06, -0.035, -0.01)
pos_point(point_h4, 0.062, -0.01, -0.01)
pos_point(point_h5, 0.065, 0.02, -0.01)
pos_point(point_h6, 0.062, 0.05, -0.01)
pos_line(line_h1, point_h1, point_h2)
pos_line(line_h2, point_h2, point_h3)
pos_line(line_h3, point_h3, point_h4)
pos_line(line_h4, point_h4, point_h5)
pos_line(line_h5, point_h5, point_h6)
pos_line(line_h6, point_h6, point_h1)


# Definition of the visual elements for fingers and fingertips
marker_f1, = ax.plot([], [], [], 'go', markersize=5)
marker_f2, = ax.plot([], [], [], 'mo', markersize=5)
marker_f3, = ax.plot([], [], [], 'co', markersize=5)
marker_f4, = ax.plot([], [], [], 'ro', markersize=5)
marker_f5, = ax.plot([], [], [], 'yo', markersize=5)
line_f1, = ax.plot([], [], [], 'k-', linewidth=2)
line_f2, = ax.plot([], [], [], 'k-', linewidth=2)   
line_f3, = ax.plot([], [], [], 'k-', linewidth=2)
line_f4, = ax.plot([], [], [], 'k-', linewidth=2)
line_f5, = ax.plot([], [], [], 'k-', linewidth=2)


# Define the relative fingers position with respect to the hand frame
# These are the initial positions of the fingers in the hand frame
# It depends on the subject's hand initial position
# Used to avoid the problem of the fingers jumping from one position to another when the real animation starts
f1_rel_prev = np.array([0.13868391, -0.05139445,  0.00857016, 1])
f2_rel_prev = np.array([0.11976245,  0.13734637, -0.00763299, 1])
f3_rel_prev = np.array([-0.0369033, 0.15533912, -0.02849389, 1])
f4_rel_prev = np.array([0.16185893, 0.08565602, 0.008876, 1])
f5_rel_prev = np.array([0.16463813, 0.03239017, 0.01738579, 1])
m_nulla = np.array([[1,0,0],[0,1,0],[0,0,1]])


#-----------------------------------------------------------------------
# Animation loop initialization - Hand kept still
#-----------------------------------------------------------------------
# Loop to keep the hand model still as the code starts the animation
for i in range(20):

    # Update finger marker positions
    pos_point(marker_f1, f1_rel_prev[0], f1_rel_prev[1], f1_rel_prev[2])
    pos_point(marker_f2, f2_rel_prev[0], f2_rel_prev[1], f2_rel_prev[2])
    pos_point(marker_f3, f3_rel_prev[0], f3_rel_prev[1], f3_rel_prev[2])
    pos_point(marker_f4, f4_rel_prev[0], f4_rel_prev[1], f4_rel_prev[2])
    pos_point(marker_f5, f5_rel_prev[0], f5_rel_prev[1], f5_rel_prev[2])

    pos_line(line_f1, point_h3, marker_f1)
    pos_line(line_f2, point_h4, marker_f5)
    pos_line(line_f3, point_h5, marker_f2)
    pos_line(line_f4, point_h6, marker_f4)
    pos_line(line_f5, point_h1, marker_f3)


    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.05)

#-----------------------------------------------------------------------
# Animation loop - Hand in motion
#-----------------------------------------------------------------------
for i in range(len(pos_hand)):
    
    # Loading of the current position of the hand and fingers wrt the world frame
    x1, y1, z1 = pos_hand[i]
    x_f1, y_f1, z_f1 = pos_f1[i]
    x_f2, y_f2, z_f2 = pos_f2[i]
    x_f3, y_f3, z_f3 = pos_f3[i]
    x_f4, y_f4, z_f4 = pos_f4[i]
    x_f5, y_f5, z_f5 = pos_f5[i]


    # Rotation matrix calculation---------------------
    # to pass from the world frame to the hand frame
    rotation_matrix = from_q_to_rotation(rot_hand[i])
    if(rotation_matrix == m_nulla).all():
        continue

    # Homogeneous transformation matrix---------------------
    # to make the base frame pass from the world frame to the hand frame
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = np.array([x1, y1, z1])


    # Calculate the relative position of the fingers in the hand frame---------------------
    f1 = np.linalg.inv(T) @ np.array([x_f1, y_f1, z_f1, 1])
    f2 = np.linalg.inv(T) @ np.array([x_f2, y_f2, z_f2, 1])
    f3 = np.linalg.inv(T) @ np.array([x_f3, y_f3, z_f3, 1])
    f4 = np.linalg.inv(T) @ np.array([x_f4, y_f4, z_f4, 1])
    f5 = np.linalg.inv(T) @ np.array([x_f5, y_f5, z_f5, 1])


    # Hungarian algorithm---------------------
    # To match the current finger positions with the previous ones
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

    # Update fingertips (marker) positions---------------------
    pos_point(marker_f1, f1_rel[0], f1_rel[1], f1_rel[2])
    pos_point(marker_f4, f2_rel[0], f2_rel[1], f2_rel[2])
    pos_point(marker_f3, f3_rel[0], f3_rel[1], f3_rel[2])
    pos_point(marker_f2, f4_rel[0], f4_rel[1], f4_rel[2])
    pos_point(marker_f5, f5_rel[0], f5_rel[1], f5_rel[2])

    # Update finger position---------------------
    pos_line(line_f1, point_h3, marker_f1)
    pos_line(line_f2, point_h4, marker_f5)
    pos_line(line_f3, point_h5, marker_f2)
    pos_line(line_f4, point_h6, marker_f4)
    pos_line(line_f5, point_h1, marker_f3)


    # Update angle of closure of the fingers---------------------

    # Get angle of closure of finger 1
    x_h3, y_h3, z_h3 = point_h3.get_data_3d()
    f1_coord = np.array([f1_rel[0], f1_rel[1], f1_rel[2]])
    h3_coord = np.array([x_h3[0], y_h3[0], z_h3[0]])
    knocle1_axis = [x_h3[0] + 0.1, y_h3[0], -0.01] - h3_coord
    knocle1_axis = knocle1_axis / np.linalg.norm(knocle1_axis)
    f1_axis = f1_coord - h3_coord
    f1_axis = f1_axis / np.linalg.norm(f1_axis)
    angle_f1 = angle_between_vectors(knocle1_axis, f1_axis)
    
    # Get angle of closure of finger 2
    x_h4, y_h4, z_h4 = point_h4.get_data_3d()
    f2_coord = np.array([f5_rel[0], f5_rel[1], f5_rel[2]])
    h4_coord = np.array([x_h4[0], y_h4[0], z_h4[0]])
    knocle2_axis = [x_h4[0] + 0.01, y_h4[0], -0.01] - h4_coord
    knocle2_axis = knocle2_axis / np.linalg.norm(knocle2_axis)
    f2_axis = f2_coord - h4_coord
    f2_axis = f2_axis / np.linalg.norm(f2_axis)
    angle_f2 = angle_between_vectors(knocle2_axis, f2_axis)

    # Get angle of closure of finger 3
    x_h5, y_h5, z_h5 = point_h5.get_data_3d()
    f3_coord = np.array([f4_rel[0], f4_rel[1], f4_rel[2]])
    h5_coord = np.array([x_h5[0], y_h5[0], z_h5[0]])
    knocle3_axis = [x_h5[0] + 0.01, y_h5[0], -0.01] - h5_coord
    knocle3_axis = knocle3_axis / np.linalg.norm(knocle3_axis)
    f3_axis = f3_coord - h5_coord
    f3_axis = f3_axis / np.linalg.norm(f3_axis)
    angle_f3 = angle_between_vectors(knocle3_axis, f3_axis)

    # Get angle of closure of finger 4
    x_h6, y_h6, z_h6 = point_h6.get_data_3d()
    f4_coord = np.array([f2_rel[0], f2_rel[1], f2_rel[2]])
    h6_coord = np.array([x_h6[0], y_h6[0], z_h6[0]])
    knocle4_axis = [x_h6[0] + 0.01, y_h6[0], -0.01] - h6_coord
    knocle4_axis = knocle4_axis / np.linalg.norm(knocle4_axis)
    f4_axis = f4_coord - h6_coord
    f4_axis = f4_axis / np.linalg.norm(f4_axis)
    angle_f4 = angle_between_vectors(knocle4_axis, f4_axis)

    # Get angle of closure of finger 5
    x_h1, y_h1, z_h1 = point_h1.get_data_3d()
    f5_coord = np.array([f3_rel[0], f3_rel[1], f3_rel[2]])
    h1_coord = np.array([x_h1[0], y_h1[0], z_h1[0]])
    knocle5_axis = [x_h1[0] + 0.01, y_h1[0] + 0.01, -0.01] - h1_coord
    knocle5_axis = knocle5_axis / np.linalg.norm(knocle5_axis)
    f5_axis = f5_coord - h1_coord
    f5_axis = f5_axis / np.linalg.norm(f5_axis)
    angle_f5 = angle_between_vectors(knocle5_axis, f5_axis)


    # Update sigma matrix value for the current frame---------------------
    # Here all the angles are normalized to express the sigma matrix value for understanding the closure of the hand
    sigma_value = normalize_angle(angle_f1, angle_f2, angle_f3, angle_f4, angle_f5)
    for i in range(n_of_times):                 # Every value is added n_of_times to the sigma list, to match the EMG data length
        sigma_vicon.append(sigma_value)


    # Save the position of the hand to use the Hungarian algorithm in the next iteration---------------------
    f1_rel_prev = f1_rel
    f2_rel_prev = f2_rel
    f3_rel_prev = f3_rel
    f4_rel_prev = f4_rel
    f5_rel_prev = f5_rel

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.001)

plt.ioff()
plt.show()


#------------------------------------------------------------------------
# Sigma matrix processing
#------------------------------------------------------------------------

# Ensure the sigma list has the same number of elements as the EMG data
sigma_len = len(sigma_vicon)
if sigma_len < timestamps_emg:
    for i in range(timestamps_emg - sigma_len):
        sigma_vicon.append(sigma_value)

if sigma_len > timestamps_emg:
    sigma = sigma_vicon[:timestamps_emg]

print(len(sigma), "sigma values generated")
print(timestamps_emg, "samples in EMG data")

# Plot the sigma values
plt.figure(figsize=(10, 3))
plt.plot(sigma, label='Sigma', color='orange')
plt.legend()
plt.show()
