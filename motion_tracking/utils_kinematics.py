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
