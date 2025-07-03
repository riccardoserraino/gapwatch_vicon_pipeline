from config import *

###########################################################################################################
# Loading purpose functions
###########################################################################################################

#----------------------------------------------------------------------------------------------------------
# Function to load EMG data from a bag file
def load_emg_data(file_path, topic='/emg'):
    """
    Load EMG data from a CSV file.
    
    Parameters:
    - file_path: str, path to the CSV file containing EMG data.
    - topic: str, the topic from which to extract EMG data (default is '/emg').

    
    Returns:
    - emg_data: np.ndarray, array of EMG data.

    """
    emg_data = []
    timestamps = []

    print("\nLoading GapWatch data...")
    # Open the bag and extract EMG values from messages
    with rosbag.Bag(file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                for i in msg.emg:  # Read each value in the EMG array
                    emg_data.append(i)
                    timestamps.append(t.to_sec())
            except AttributeError as e:
                print("Message missing expected fields:", e)
                break

    emg_data = np.array(emg_data)
    timestamps = np.array(timestamps)
    print("Loading GapWatch data completed.\n")

    return emg_data, timestamps


#----------------------------------------------------------------------------------------------------------
# Function to reshape the original data loaded
# The bag file streams data as a flat list and we need 16 channels shape
def reshape_emg_data(emg_data, timestamps):
    """
    Reshape the EMG data into a 16-channel matrix and calculate timestamps and sampling frequency

    Parameters:
    - emg_data: np.ndarray, array of EMG data.
    - timestamps: np.ndarray, array of timestamps corresponding to the EMG data.

    Returns:
    - raw_emg: np.ndarray, reshaped EMG data in a 16-channel format.
    - reshaped_timestamps: np.ndarray, timestamps reshaped to match the EMG data
    - fs: float, calculated sampling frequency of the EMG data.
    """

    # SIgnal reshaping
    selector = 0
    raw_emg = np.empty((16, 0))  # Initialize empty matrix with 16 rows (channels)
    # Loop over all complete sets of 16-channel samples
    print("\nReshaping GapWatch data into 16-channel matrix...")
    for i in range(int(len(emg_data)/16)):
        temp = emg_data[selector:selector+16]           # Extract 16 consecutive samples
        new_column = np.array(temp).reshape(16, 1)      # Convert to column format
        raw_emg = np.hstack((raw_emg, new_column))      # Append column to EMG matrix
        selector += 16                                  # Move to next block
        #print("Sample number: ", i)
    print("Reshaping GapWatch data completed.\n")

    # Timestamps reshaping & frequency calculation
    reshaped_timestamps = timestamps[::16]                  
    reshaped_timestamps_int = len(reshaped_timestamps)     
    duration_c = reshaped_timestamps[-1] - reshaped_timestamps[0]
    fs=reshaped_timestamps_int/duration_c

    # Print shape information of calibration data
    print("\nInsights into final GapWatch data:")
    print(f" - Acquired EMG data shape: {emg_data.shape}")  # Should be (n_samples * 16, )
    print(f" - Reshaped EMG shape: {raw_emg.shape}")        # Should be (n_samples, n_channels)
    print(f" - EMG Timestamps count: {reshaped_timestamps_int}")# Should be (n_samples)
    print(f" - Duration of EMG data: {duration_c:.2f} s")
    print(f" - Sampling frequency fs of EMG data : {fs:.2f} Hz\n")

    return raw_emg, reshaped_timestamps_int, fs


#----------------------------------------------------------------------------------------------------------
# Function to load Vicon data from a bag file
def load_vicon_data(file_path, topic_hand='/tf', topic_marker='/vicon/unlabeled_markers'):
    """
    Load Vicon data from a bag file.

    Parameters:
        - file_path: str, path to the bag file containing Vicon data.
        - topic_hand: str, the topic from which to extract hand positions and orientations (default is '/tf').
        - topic_marker: str, the topic from which to extract marker positions (default is '/vicon/unlabeled_markers').
    
    Returns:
        - hand_positions: list of lists, each containing the [x, y, z] coordinates of the hand.
        - hand_orientations: list of lists, each containing the [x, y, z, w] quaternion orientation of the hand.
        - marker_positions: list of lists, each containing the [x, y, z] coordinates of the markers (5 markers).
        - timestamps: list of timestamps corresponding to each frame.
    """

    # Initialize lists to store hand and marker positions and orientation (only of the hand)
    hand_positions = []
    hand_orientations = []
    marker_positions = [[] for _ in range(5)]  # 5 marker
    last_marker_positions = [None] * 5         # To track last valid positions of markers (to use in case we lost a marker for a frame)
    timestamps = []                            # To track the time of each frame
    
    print("\nLoading Vicon data...")
    # Open the bag and extract hand and marker positions from the bag file
    with rosbag.Bag(file_path, 'r') as bag:

        for topic, msg, t in bag.read_messages():
            if topic == topic_hand:    # hand positions and orientations topic
                for transform in msg.transforms:
                    if transform.child_frame_id == "hand":
                        pos = transform.transform.translation
                        rot = transform.transform.rotation
                        hand_positions.append([pos.x, pos.y, pos.z])
                        hand_orientations.append([rot.x, rot.y, rot.z, rot.w])
                        timestamps.append(t.to_sec())

            elif topic == topic_marker:    # marker positions topic
                current_marker_positions = [None] * 5
                for i, marker in enumerate(msg.markers):     # Loop to update the last known positions of the markers
                    if i < 5:
                        pos = marker.pose.position
                        current_marker_positions[i] = [pos.x, pos.y, pos.z]
                        last_marker_positions[i] = current_marker_positions[i]  

                for i in range(5):                                     # Loop to append the current marker positions to the marker_positions list
                    if current_marker_positions[i] is None:            # If the marker is not found in the current frame, use the last known position
                        if last_marker_positions[i] is not None:
                            marker_positions[i].append(last_marker_positions[i])
                        else:
                            marker_positions[i].append([np.nan, np.nan, np.nan])  # If no last position is available, append NaN
                    else:
                        marker_positions[i].append(current_marker_positions[i])

    print("Loading Vicon data completed.\n")

    print("\nInsights into final Vicon data:")
    print(f" - Hand positions count: {len(hand_positions)}")  # Number of hand positions
    print(f" - Hand orientations count: {len(hand_orientations)}")  
    for i, positions in enumerate(marker_positions):
        print(f" - Marker {i} positions count: {len(positions)}")
    print(f" - Timestamps Vicon count: {len(timestamps)}")  # Number of timestamps
    print(f" - Duration of Vicon data: {timestamps[-1] - timestamps[0]:.2f} s\n")  # Duration of the Vicon data

    return hand_positions, hand_orientations, marker_positions, timestamps


#----------------------------------------------------------------------------------------------------------

