from config import *

###########################################################################################################
# Kinematics & Optimization helper functions
###########################################################################################################

#----------------------------------------------------------------------------------------------------------
# Function to get the homogeneous transformation matrix of a knuckle
def get_htm_mp(knucle, target, target_o=0):
    x1, y1, z1 = knucle.get_data_3d()
    p1 = np.array([x1[0], y1[0], z1[0]])
    x_axis = target - p1
    '''
    if (math.sqrt(x_axis[0]**2 + x_axis[1]**2) < 0.035):
        x_axis = target_o - p1
    '''
    x_axis = x_axis / np.linalg.norm(x_axis)
    temp_vec = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(x_axis, temp_vec)), 1.0):
        temp_vec = np.array([0, 1, 0])
    z_axis = np.cross(x_axis, temp_vec)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    translation = [x1[0], y1[0], z1[0]]
    Homogeneous_tm = np.eye(4)
    Homogeneous_tm[:3, :3] = rotation
    Homogeneous_tm[:3, 3] = translation
    return Homogeneous_tm


#----------------------------------------------------------------------------------------------------------
# Function to calculate the inverse kinematic of the hand
def inverse_kinematic(final_point, l1, l2, l3, phi=140, threshold=1):
    x_p = final_point[0] - l3 * math.cos(math.radians(phi))
    y_p = final_point[1] - l3 * math.sin(math.radians(phi))
    theta_2 = - math.degrees(math.acos((x_p**2 + y_p**2 - l1**2 - l2**2)/(2*l1*l2)))
    k1 = l1 + l2 * math.cos(math.radians(theta_2))
    k2 = l2 * math.sin(math.radians(theta_2))
    sin_theta_1 = (y_p*k1 - x_p*k2)/(k1**2+k2**2)
    cos_theta_1 = (y_p - k1*sin_theta_1)/k2
    theta_1 = math.degrees(math.atan2(sin_theta_1, cos_theta_1))
    theta_3 = phi - theta_1 - theta_2
    if (theta_3 > threshold):
        raise Exception("Messaggio di errore")
    return theta_1, theta_2, theta_3


#----------------------------------------------------------------------------------------------------------
# Function to calculate the direct kinematic of the hand
def direct_kinematic(theta1, theta2, theta3, l1, l2, l3):
    x0, y0 = 0, 0
    x1 = x0 + l1 * math.cos(math.radians(theta1))
    y1 = y0 + l1 * math.sin(math.radians(theta1))
    x2 = x1 + l2 * math.cos(math.radians(theta1 + theta2))
    y2 = y1 + l2 * math.sin(math.radians(theta1 + theta2))
    x3 = x2 + l3 * math.cos(math.radians(theta1 + theta2 + theta3))
    y3 = y2 + l3 * math.sin(math.radians(theta1 + theta2 + theta3))
    return x1, y1, x2, y2, x3, y3


#----------------------------------------------------------------------------------------------------------
# Function to convert final position to joint positions
def from_final_pos_to_joints(T, final_pos, l1, l2, l3, phi, threshold=10):
    f_in_knucle_coo = np.linalg.inv(T) @ np.array([final_pos[0], final_pos[1], final_pos[2], 1])
    theta_1, theta_2, theta_3 = inverse_kinematic(f_in_knucle_coo[:2], l1, l2, l3, phi, threshold)
    x1, y1, x2, y2, x3, y3 = direct_kinematic(theta_1, theta_2, theta_3, l1, l2, l3)
    joint1_pos = [x1, y1, 0, 1]
    joint2_pos = [x2, y2, 0, 1]
    joint1_o = T @ np.array(joint1_pos)
    joint2_o = T @ np.array(joint2_pos)
    return joint1_o, joint2_o


#----------------------------------------------------------------------------------------------------------
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
# Function to show the frame of a joint
def print_axis(ax, point_1, point_2):
    x1, y1, z1 = point_1.get_data_3d()
    p1 = np.array([x1[0], y1[0], z1[0]])
    p2 = point_2
    x_axis = p2 - p1
    x_axis = x_axis / np.linalg.norm(x_axis)
    temp_vec = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(x_axis, temp_vec)), 1.0):
        temp_vec = np.array([0, 1, 0])
    z_axis = np.cross(x_axis, temp_vec)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    arrow_x = ax.quiver(*p1, *(x_axis * 0.05), color='r', linewidth=2)
    arrow_y = ax.quiver(*p1 , *(y_axis * 0.05), color='g', linewidth=2)
    arrow_z = ax.quiver(*p1 , *(z_axis * 0.05), color='b', linewidth=2)
    return x_axis, y_axis, z_axis


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
