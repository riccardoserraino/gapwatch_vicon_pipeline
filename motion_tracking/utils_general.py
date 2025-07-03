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
def calculate_finger_angle(finger_pos, hand_point, axis_offset):
    """
    
    """
    
    hand_coord = hand_point[:3]
    finger_coord = finger_pos[:3]
    
    knuckle_axis = (hand_coord + axis_offset) - hand_coord
    knuckle_axis = knuckle_axis / np.linalg.norm(knuckle_axis)
    
    finger_axis = finger_coord - hand_coord
    finger_axis = finger_axis / np.linalg.norm(finger_axis)
    
    return angle_between_vectors(knuckle_axis, finger_axis)

#----------------------------------------------------------------------------------------------------------
