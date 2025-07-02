from config import *


###########################################################################################################
# General purpose functions
###########################################################################################################

#----------------------------------------------------------------------------------------------------------
# Function to see if the quaternion is valid
def is_quaternion_valid(q):
    return all(not math.isnan(v) for v in [q[0], q[1], q[2], q[3]])


#----------------------------------------------------------------------------------------------------------
# Function to convert a quaternion into a rotation matrix
def from_q_to_rotation(q):
    if not is_quaternion_valid(q):
        q[0], q[1], q[2], q[3] = 0, 0, 0, 1
        rotation_matrix = R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()

    else:
        rotation_matrix = R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
    return rotation_matrix


#----------------------------------------------------------------------------------------------------------
# Function to assign position to lines
def pos_line(line, point_1, point_2):
    x1, y1, z1 = point_1.get_data_3d()
    x2, y2, z2 = point_2.get_data_3d()
    x_data = [x1[0], x2[0]]
    y_data = [y1[0], y2[0]]
    z_data = [z1[0], z2[0]]
    line.set_data(x_data, y_data)
    line.set_3d_properties(z_data)


#----------------------------------------------------------------------------------------------------------
# Function to assign coordinates to points
def pos_point(marker, x, y, z):
    marker.set_data([x], [y])
    marker.set_3d_properties([z])


#----------------------------------------------------------------------------------------------------------
# Function to calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)  # protezione numerica
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


#----------------------------------------------------------------------------------------------------------
# Function to give the closure value of the hand
def normalize_angle(angle_f1, angle_f2, angle_f3, angle_f4, angle_f5):
    closure_ang1 = ((angle_f1 - 7.53)*0.2)/(103.93 - 7.53)
    closure_ang2 = ((angle_f2 - 19.8)*0.2)/(119.67 - 19.8)
    closure_ang3 = ((angle_f3 - 22.43)*0.2)/(126.41 - 22.43)
    closure_ang4 = ((angle_f4 - 46.4)*0.2)/(122.4 - 46.4)
    closure_ang5 = ((angle_f5 - 51.27)*0.2)/(73.18 - 51.27)
    normalized_angles = [closure_ang1, closure_ang2, closure_ang3, closure_ang4, closure_ang5]
    for i in range(len(normalized_angles)):
        if normalized_angles[i] < 0:
            normalized_angles[i] = 0
        elif normalized_angles[i] > 1:
            normalized_angles[i] = 0.2
    closure = closure_ang1 + closure_ang2 + closure_ang3 + closure_ang4 + closure_ang5
    return closure


#----------------------------------------------------------------------------------------------------------
# Compute the angle of closure for a given finger and knuckle.
def compute_finger_angle(finger_rel, hand_point):
    """
    Computes the angle between a finger and the knuckle axis of the hand.

    Parameters:
        finger_rel: np.ndarray, the relative position of the finger in 3D space.
        hand_point: object, the point representing the hand in 3D space.

    Returns:
        float: the angle in degrees between the finger and the knuckle axis.
    """

    x_h, y_h, z_h = hand_point.get_data_3d()
    
    finger_coord = np.array([finger_rel[0], finger_rel[1], finger_rel[2]])
    hand_coord = np.array([x_h[0], y_h[0], z_h[0]])

    knuckle_axis = np.array([x_h[0] + 0.1, y_h[0], -0.01]) - hand_coord

    knuckle_axis = knuckle_axis / np.linalg.norm(knuckle_axis)
    finger_axis = finger_coord - hand_coord
    finger_axis = finger_axis / np.linalg.norm(finger_axis)
    
    return angle_between_vectors(knuckle_axis, finger_axis)


#----------------------------------------------------------------------------------------------------------
# Function to update finger marker and connecting line.
def update_finger_visual(marker, line, hand_point, finger_rel):
    """
    Updates the position of a finger marker and its connecting line to the hand point.
    Parameters:
        marker: object, the marker representing the finger in 3D space.
        line: object, the line connecting the hand point to the finger marker.
        hand_point: object, the point representing the hand in 3D space.
        finger_rel: np.ndarray, the relative position of the finger in 3D space.
    """

    pos_point(marker, finger_rel[0], finger_rel[1], finger_rel[2])
    pos_line(line, hand_point, marker)