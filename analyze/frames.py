from config import *

def plot_reference_frame(ax, origin, rotation_matrix, scale=1.0, label=''):
    """
    Plots a 3D reference frame (origin and axes).

    Args:
        ax: Matplotlib 3D axes object.
        origin (np.array): 3D coordinates of the frame's origin.
        rotation_matrix (np.array): 3x3 rotation matrix for the frame's orientation.
        scale (float): Length of the axes.
        label (str): Label for the frame.
    """
    x_axis = rotation_matrix @ np.array([scale, 0, 0])
    y_axis = rotation_matrix @ np.array([0, scale, 0])
    z_axis = rotation_matrix @ np.array([0, 0, scale])

    # Plot origin
    ax.scatter(*origin, color='k', marker='o', s=50, label=f'{label} Origin' if label else '')

    # Plot axes
    ax.quiver(*origin, *x_axis, color='r', length=1, arrow_length_ratio=0.1, label=f'{label} X-axis' if label else '')
    ax.quiver(*origin, *y_axis, color='g', length=1, arrow_length_ratio=0.1, label=f'{label} Y-axis' if label else '')
    ax.quiver(*origin, *z_axis, color='b', length=1, arrow_length_ratio=0.1, label=f'{label} Z-axis' if label else '')

    # Add text labels for axes
    ax.text(*(origin + x_axis * 1.1), f'{label}X', color='r')
    ax.text(*(origin + y_axis * 1.1), f'{label}Y', color='g')
    ax.text(*(origin + z_axis * 1.1), f'{label}Z', color='b')


def rotation_matrix_z(theta_deg):
    """Generates a 3x3 rotation matrix around the Z-axis."""
    theta_rad = np.radians(theta_deg)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])

def rotation_matrix_y(theta_deg):
    """Generates a 3x3 rotation matrix around the Y-axis."""
    theta_rad = np.radians(theta_deg)
    return np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

def rotation_matrix_x(theta_deg):
    """Generates a 3x3 rotation matrix around the X-axis."""
    theta_rad = np.radians(theta_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])

# 1. Setup the 3D plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Vicon-World Reference Frame with Hand Frame and Fingertip Markers')
ax.set_aspect('equal', adjustable='box') # Ensures proper aspect ratio

# Set limits for better visualization
plot_range = 2
ax.set_xlim([-plot_range, plot_range])
ax.set_ylim([-plot_range, plot_range])
ax.set_zlim([-0.5, plot_range * 1.5])

# 2. Define the World Reference Frame
world_origin = np.array([0, 0, 0])
world_rotation_matrix = np.eye(3)  # Identity matrix for no rotation (aligned with global axes)
plot_reference_frame(ax, world_origin, world_rotation_matrix, scale=0.8, label='Vicon')

# 3. Define the Hand Reference Frame
# Translation: Placed "above" the world frame (positive Z) and slightly offset
hand_translation = np.array([0.5, -0.4, 1.2]) # x, y, z translation

# Rotation: Rotate the hand frame. Let's rotate around Z and then Y for a combined effect.
rotation_x = rotation_matrix_x(90)
rotation_z = rotation_matrix_z(-20)
rotation_y = rotation_matrix_y(20)
hand_rotation_matrix = rotation_x @ rotation_y @ rotation_z # Order matters for combined rotations

# Origin of the hand frame relative to the world frame
hand_origin = world_origin + hand_translation

plot_reference_frame(ax, hand_origin, hand_rotation_matrix, scale=0.6, label='Hand')

# Add a connecting line from world origin to hand origin (optional, for clarity)
ax.plot([world_origin[0], hand_origin[0]],
        [world_origin[1], hand_origin[1]],
        [world_origin[2], hand_origin[2]],
        'k--', linewidth=0.5)


# 4. Define and plot 5 markers for fingertips in the Hand Frame
# These points are relative to the hand's origin (0,0,0) in its OWN coordinate system
fingertip_local_positions = np.array([
    [0.13, -0.12, 0.3],   # Thumb (slightly different X/Y)
    [0.05, -0.15, 0.4], # Index finger
    [0.0, -0.16, 0.42], # Middle finger
    [-0.05, -0.15, 0.4],# Ring finger
    [-0.1, -0.12, 0.35] # Pinky finger
])
# Scale these positions to be reasonable relative to the hand frame's size
fingertip_local_positions *= -1.0 # Scale for visualization purposes

# Transform local fingertip positions to world coordinates
fingertip_world_positions = []
for local_pos in fingertip_local_positions:
    # Rotate the local position vector
    rotated_pos = hand_rotation_matrix @ local_pos
    # Translate it by the hand frame's origin
    world_pos = hand_origin + rotated_pos
    fingertip_world_positions.append(world_pos)

fingertip_world_positions = np.array(fingertip_world_positions)

# Plot the markers
ax.scatter(fingertip_world_positions[:, 0],
           fingertip_world_positions[:, 1],
           fingertip_world_positions[:, 2],
           color='m',  # Magenta color for markers
           marker='o', # Circle marker
           s=10,      # Size of the markers
           label='Fingertip Markers')

plt.grid(True)
plt.show()