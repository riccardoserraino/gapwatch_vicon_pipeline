from config import *

###########################################################################################################
# Visualize helper functions
###########################################################################################################

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


