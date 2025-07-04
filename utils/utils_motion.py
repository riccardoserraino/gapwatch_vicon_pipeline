from config import *

###########################################################################################################
# Kinematics & Optimization helper functions
###########################################################################################################

#----------------------------------------------------------------------------------------------------------
def get_htm_mp(knucle, target, target_o=0):
    """
    Computes the homogeneous transformation matrix for a knuckle point relative to a target.
    
    Parameters:
        knucle: Knuckle object containing 3D position data
        target: Target point coordinates as numpy array [x,y,z]
        target_o: Optional offset for target (default=0)
    
    Returns:
        Homogeneous_tm: 4x4 homogeneous transformation matrix
    """
    # Extract knuckle position data
    x1, y1, z1 = knucle.get_data_3d()
    p1 = np.array([x1[0], y1[0], z1[0]])
    
    # Calculate x-axis (primary axis) from knuckle to target
    x_axis = target - p1
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Create temporary vector for axis calculation
    temp_vec = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(x_axis, temp_vec)), 1.0):
        temp_vec = np.array([0, 1, 0])
    
    # Calculate z-axis (third axis) via cross product
    z_axis = np.cross(x_axis, temp_vec)
    z_axis /= np.linalg.norm(z_axis)
    
    # Calculate y-axis (second axis) via cross product
    y_axis = np.cross(z_axis, x_axis)
    
    # Construct rotation matrix from axes
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    
    # Create homogeneous transformation matrix
    translation = [x1[0], y1[0], z1[0]]
    Homogeneous_tm = np.eye(4)
    Homogeneous_tm[:3, :3] = rotation
    Homogeneous_tm[:3, 3] = translation
    
    return Homogeneous_tm


#----------------------------------------------------------------------------------------------------------
def inverse_kinematic(final_point, l1, l2, l3, phi=140, threshold=1):
    """
    Calculates inverse kinematics for a 3-link planar manipulator.
    
    Parameters:
        final_point: Target point coordinates [x,y]
        l1, l2, l3: Lengths of the three links
        phi: Total angle sum constraint in degrees (default=140)
        threshold: Maximum allowed error for theta3 (default=1)
    
    Returns:
        theta_1, theta_2, theta_3: Joint angles in degrees
    
    Raises:
        Exception if theta3 exceeds threshold
    """
    # Calculate intermediate point based on phi angle
    x_p = final_point[0] - l3 * math.cos(math.radians(phi))
    y_p = final_point[1] - l3 * math.sin(math.radians(phi))
    
    # Calculate theta2 using law of cosines
    theta_2 = - math.degrees(math.acos((x_p**2 + y_p**2 - l1**2 - l2**2)/(2*l1*l2)))
    
    # Calculate intermediate terms for theta1
    k1 = l1 + l2 * math.cos(math.radians(theta_2))
    k2 = l2 * math.sin(math.radians(theta_2))
    
    # Calculate theta1 components
    sin_theta_1 = (y_p*k1 - x_p*k2)/(k1**2+k2**2)
    cos_theta_1 = (y_p - k1*sin_theta_1)/k2
    theta_1 = math.degrees(math.atan2(sin_theta_1, cos_theta_1))
    
    # Calculate theta3 based on angle sum constraint
    theta_3 = phi - theta_1 - theta_2
    
    # Validate theta3 against threshold
    if (theta_3 > threshold):
        raise Exception("Messaggio di errore")
    
    return theta_1, theta_2, theta_3


#----------------------------------------------------------------------------------------------------------
def direct_kinematic(theta1, theta2, theta3, l1, l2, l3):
    """
    Calculates forward kinematics for a 3-link planar manipulator.
    
    Parameters:
        theta1, theta2, theta3: Joint angles in degrees
        l1, l2, l3: Lengths of the three links
    
    Returns:
        x1, y1: First joint position
        x2, y2: Second joint position
        x3, y3: End effector position
    """
    # Start from origin
    x0, y0 = 0, 0
    
    # Calculate first joint position
    x1 = x0 + l1 * math.cos(math.radians(theta1))
    y1 = y0 + l1 * math.sin(math.radians(theta1))
    
    # Calculate second joint position
    x2 = x1 + l2 * math.cos(math.radians(theta1 + theta2))
    y2 = y1 + l2 * math.sin(math.radians(theta1 + theta2))
    
    # Calculate end effector position
    x3 = x2 + l3 * math.cos(math.radians(theta1 + theta2 + theta3))
    y3 = y2 + l3 * math.sin(math.radians(theta1 + theta2 + theta3))
    
    return x1, y1, x2, y2, x3, y3


#----------------------------------------------------------------------------------------------------------
def from_final_pos_to_joints(T, final_pos, l1, l2, l3, phi, threshold=10):
    """
    Converts final position to joint positions using inverse kinematics.
    
    Parameters:
        T: Transformation matrix
        final_pos: Target end effector position [x,y,z]
        l1, l2, l3: Link lengths
        phi: Angle sum constraint
        threshold: IK solution threshold (default=10)
    
    Returns:
        joint1_o, joint2_o: Joint positions in original coordinate frame
    """
    # Transform final position to knuckle coordinate frame
    f_in_knucle_coo = np.linalg.inv(T) @ np.array([final_pos[0], final_pos[1], final_pos[2], 1])
    
    # Calculate joint angles using inverse kinematics
    theta_1, theta_2, theta_3 = inverse_kinematic(f_in_knucle_coo[:2], l1, l2, l3, phi, threshold)
    
    # Calculate joint positions using forward kinematics
    x1, y1, x2, y2, x3, y3 = direct_kinematic(theta_1, theta_2, theta_3, l1, l2, l3)
    
    # Transform joint positions back to original coordinate frame
    joint1_pos = [x1, y1, 0, 1]
    joint2_pos = [x2, y2, 0, 1]
    joint1_o = T @ np.array(joint1_pos)
    joint2_o = T @ np.array(joint2_pos)
    
    return joint1_o, joint2_o


#----------------------------------------------------------------------------------------------------------
def pos_line(line, point_1, point_2):
    """
    Updates a line object with new endpoint positions.
    
    Parameters:
        line: Matplotlib line object to update
        point_1: First endpoint object with get_data_3d() method
        point_2: Second endpoint object with get_data_3d() method
    """
    # Extract coordinates from both points
    x1, y1, z1 = point_1.get_data_3d()
    x2, y2, z2 = point_2.get_data_3d()
    
    # Create line data arrays
    x_data = [x1[0], x2[0]]
    y_data = [y1[0], y2[0]]
    z_data = [z1[0], z2[0]]
    
    # Update line object
    line.set_data(x_data, y_data)
    line.set_3d_properties(z_data)


#----------------------------------------------------------------------------------------------------------
def print_axis(ax, point_1, point_2):
    """
    Draws coordinate axes at a point oriented toward another point.
    
    Parameters:
        ax: Matplotlib 3D axis object
        point_1: Origin point object with get_data_3d() method
        point_2: Target point coordinates [x,y,z]
    
    Returns:
        x_axis, y_axis, z_axis: The calculated axis vectors
    """
    # Get origin point coordinates
    x1, y1, z1 = point_1.get_data_3d()
    p1 = np.array([x1[0], y1[0], z1[0]])
    p2 = point_2
    
    # Calculate and normalize x-axis
    x_axis = p2 - p1
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Create temporary vector for axis calculation
    temp_vec = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(x_axis, temp_vec)), 1.0):
        temp_vec = np.array([0, 1, 0])
    
    # Calculate and normalize z-axis
    z_axis = np.cross(x_axis, temp_vec)
    z_axis /= np.linalg.norm(z_axis)
    
    # Calculate y-axis
    y_axis = np.cross(z_axis, x_axis)
    
    # Draw axis arrows
    arrow_x = ax.quiver(*p1, *(x_axis * 0.05), color='r', linewidth=2)
    arrow_y = ax.quiver(*p1 , *(y_axis * 0.05), color='g', linewidth=2)
    arrow_z = ax.quiver(*p1 , *(z_axis * 0.05), color='b', linewidth=2)
    
    return x_axis, y_axis, z_axis


#----------------------------------------------------------------------------------------------------------
# Function to see if the quaternion is valid
def is_quaternion_valid(q):
    """
    Checks if a quaternion contains valid (non-NaN) values.
    
    Parameters:
        q: Quaternion as [x,y,z,w]
    
    Returns:
        Boolean indicating validity
    """
    return all(not math.isnan(v) for v in [q[0], q[1], q[2], q[3]])


#----------------------------------------------------------------------------------------------------------
def from_q_to_rotation(q):
    """
    Converts a quaternion to a rotation matrix.
    
    Parameters:
        q: Quaternion as [x,y,z,w]
    
    Returns:
        rotation_matrix: 3x3 rotation matrix
    """
    # Handle invalid quaternion case
    if not is_quaternion_valid(q):
        q[0], q[1], q[2], q[3] = 0, 0, 0, 1
    
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
    
    return rotation_matrix


#----------------------------------------------------------------------------------------------------------
def pos_line(line, point_1, point_2):
    """
    Updates the position of a 3D line object between two points.
    
    Parameters:
        line (matplotlib.lines.Line3D): The line object to be updated
        point_1: First endpoint object with get_data_3d() method
                 Expected to return (x_coords, y_coords, z_coords)
        point_2: Second endpoint object with get_data_3d() method
                 Expected to return (x_coords, y_coords, z_coords)
    
    Returns:
        None: The input line object is modified in-place
    """
    # Extract 3D coordinates from both points
    # Each get_data_3d() call returns (x_coords, y_coords, z_coords)
    # We take the first element [0] from each coordinate array
    x1, y1, z1 = point_1.get_data_3d()
    x2, y2, z2 = point_2.get_data_3d()
    
    # Create line data arrays using the first elements of the coordinate arrays
    x_data = [x1[0], x2[0]]  # X-coordinates for both endpoints
    y_data = [y1[0], y2[0]]  # Y-coordinates for both endpoints
    z_data = [z1[0], z2[0]]  # Z-coordinates for both endpoints
    
    # Update the line object with new coordinates
    line.set_data(x_data, y_data)         # Set 2D data (x and y coordinates)
    line.set_3d_properties(z_data)        # Set z-coordinates for 3D plot


#----------------------------------------------------------------------------------------------------------
def pos_point(marker, x, y, z):
    """
    Updates a marker's position in 3D space.
    
    Parameters:
        marker: Matplotlib marker object
        x, y, z: New position coordinates
    """
    marker.set_data([x], [y])
    marker.set_3d_properties([z])


#----------------------------------------------------------------------------------------------------------
def angle_between_vectors(v1, v2):
    """
    Calculates the angle between two vectors in degrees.
    
    Parameters:
        v1, v2: Input vectors
    
    Returns:
        angle_deg: Angle between vectors in degrees
    """
    # Normalize input vectors
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    
    # Calculate dot product with numerical protection
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


#----------------------------------------------------------------------------------------------------------
def normalize_angle(angle_f1, angle_f2, angle_f3, angle_f4, angle_f5):
    """
    Normalizes finger angles to a 0-0.2 range and sums them for closure value.
    
    Parameters:
        angle_f1 to angle_f5: Angles for each finger
    
    Returns:
        closure: Sum of normalized angles (0-1 range)
    """
    
    # Normalize each finger angle to 0-0.2 range
    closure_ang1 = ((angle_f1 - 1.7121306263980842)*0.2)/(104.01385158747752 - 1.7121306263980842)
    closure_ang2 = ((angle_f2 - 0.07957380063842415)*0.2)/(113.15397453628636 - 0.07957380063842415)
    closure_ang3 = ((angle_f3 - 0.04798790243154717)*0.2)/(114.90405817969317 - 0.04798790243154717)
    closure_ang4 = ((angle_f4 - 10.136765578461862)*0.2)/(112.90098354889193 - 10.136765578461862)
    closure_ang5 = ((angle_f5 - 19.37360064478024)*0.2)/(71.02221487695077 - 19.37360064478024)
    
    # Create list of normalized angles
    normalized_angles = [closure_ang1, closure_ang2, closure_ang3, closure_ang4, closure_ang5]
    
    # Clip values to valid range
    for i in range(len(normalized_angles)):
        if normalized_angles[i] < 0:
            normalized_angles[i] = 0
        elif normalized_angles[i] > 1:
            normalized_angles[i] = 0.2
    
    # Calculate total closure value
    closure = closure_ang1 + closure_ang2 + closure_ang3 + closure_ang4 + closure_ang5
    
    return closure


#----------------------------------------------------------------------------------------------------------
def calculate_marker_angle(finger_pos, hand_point, axis_offset):
    """
    Calculates finger angle relative to hand coordinate system. (model with no joints)
    
    Parameters:
        finger_pos: Finger position coordinates
        hand_point: Hand reference point coordinates
        axis_offset: Offset for axis calculation
    
    Returns:
        Angle between knuckle axis and finger axis in degrees
    """
    # Extract coordinates
    hand_coord = hand_point[:3]
    finger_coord = finger_pos[:3]
    
    # Calculate knuckle axis
    knuckle_axis = (hand_coord + axis_offset) - hand_coord
    knuckle_axis = knuckle_axis / np.linalg.norm(knuckle_axis)
    
    # Calculate finger axis
    finger_axis = finger_coord - hand_coord
    finger_axis = finger_axis / np.linalg.norm(finger_axis)
    
    return angle_between_vectors(knuckle_axis, finger_axis)


#----------------------------------------------------------------------------------------------------------
def get_closure_angle(point, node, reference):
    """
    Calculates closure angle between finger joint and reference axis.
    
    Parameters:
        point: Reference point object with get_data_3d() method
        node: Finger node containing joint coordinates
        reference: Reference axis vector
    
    Returns:
        Angle between reference axis and finger joint axis in degrees
    """
    # Extract point coordinates
    ref_coord = np.array(point)
    
    # Get joint coordinates from node
    joint_coord = np.array(node[2])
    
    # Calculate and normalize reference axis
    axis = reference - ref_coord
    axis = axis / np.linalg.norm(axis)
    
    # Calculate and normalize finger axis
    finger_axis = joint_coord - ref_coord
    finger_axis = finger_axis / np.linalg.norm(finger_axis)
    
    return angle_between_vectors(axis, finger_axis)


#----------------------------------------------------------------------------------------------------------
def get_closure_angle_plot(point, node, reference):
    """
    Calculates closure angle between finger joint and reference axis, for hand model script only.
    
    Parameters:
        point: Reference point object with get_data_3d() method
        node: Finger node containing joint coordinates
        reference: Reference axis vector
    
    Returns:
        Angle between reference axis and finger joint axis in degrees
    """
    # Extract point coordinates
    x, y, z = point.get_data_3d()
    ref_coord = np.array([x[0], y[0], z[0]])
    
    # Get joint coordinates from node
    joint_coord = np.array(node[2])
    
    # Calculate and normalize reference axis
    axis = reference - ref_coord
    axis = axis / np.linalg.norm(axis)
    
    # Calculate and normalize finger axis
    finger_axis = joint_coord - ref_coord
    finger_axis = finger_axis / np.linalg.norm(finger_axis)
    
    return angle_between_vectors(axis, finger_axis)


#----------------------------------------------------------------------------------------------------------
def compute_axis(point1, point2):
    """
    Computes and returns the normalized axis vector between two 3D points.
    
    Parameters:
        point1 (np.array or list): First 3D point coordinates [x,y,z]
        point2 (np.array or list): Second 3D point coordinates [x,y,z]
    
    Returns:
        tuple: (axis_vector, unit_axis) where:
               - axis_vector: The vector from point2 to point1
               - unit_axis: The normalized (unit length) version of axis_vector
    
    Note: Currently overwrites input points with hardcoded values for testing.
    """
    
    # Temporary hardcoded values for testing/demonstration
    # (These override the function parameters - likely for debugging)
    point1 = np.array([0.087, 0.027, -0.025])  # Example point 1 coordinates
    point2 = np.array([0.07, 0.07, -0.025])   # Example point 2 coordinates
    
    # Compute the directional vector from point2 to point1
    axis_vector = point1 - point2  # Vector from point2 to point1
    
    # Calculate the unit vector (normalized version) of the axis vector
    unit_axis = axis_vector / np.linalg.norm(axis_vector)  # Normalized to length 1
    
    # Debug prints (typically removed in production code)
    #print("Axis vector:", axis_vector)  # Shows raw vector components
    #print("Normalized axis:", unit_axis)  # Shows unit vector components
    
    return axis_vector, unit_axis


#----------------------------------------------------------------------------------------------------------
