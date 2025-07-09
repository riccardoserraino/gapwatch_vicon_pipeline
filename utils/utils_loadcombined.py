from config import *

#----------------------------------------------------------------------------------------------------------
# Ask user what dataset to combine and in what order, used for multiple gesture analysis
def select_datasets(dataset_options):
    print("Select the dataset you want to combine:")

    while True:
        # Ensure input is handled robustly
        user_input = input("Enter the dataset numbers (from 1 to 9) in the desired order (e.g., '1 2 3' or '2 1'): ")
        selected_numbers_str = user_input.split()

        selected_paths = []
        all_valid = True
        for num_str in selected_numbers_str:
            if num_str in dataset_options:
                selected_paths.append(dataset_options[num_str])
            else:
                print(f"Invalid dataset number entered: {num_str}. Please use numbers from 1 to 9 corresponding to your options.")
                all_valid = False
                break # Exit the inner loop and ask for input again

        if all_valid and selected_paths:
            return selected_paths  # Return valid dataset paths
        elif all_valid and not selected_paths:
            print("No valid datasets selected. Please try again.")
        else:
            # Error message for invalid input already printed in the loop
            pass



#----------------------------------------------------------------------------------------------------------
def load_combined_reshape(selected_paths, topic='/emg'):
    
    emg_data_combined = np.empty((16, 0))
    timestamps_combined = []
    duration_combined = 0
    total_ts_reshaped = 0  


    for bag_path in selected_paths:

        emg_data = []
        timestamps = []
        

        print("\nLoading GapWatch data single...")
        # Open the bag and extract EMG values from messages
        with rosbag.Bag(bag_path, 'r') as bag:
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
        print("emg_loaded raw: ", emg_data.shape)
        print("timestamps raw: ", timestamps.shape[0])
        duration = timestamps[-1] -timestamps[0]
        print(f"duration: {duration:.2f} s\n")


        print("\nReshaping GapWatch data single...")
        # Signal reshaping
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

        reshaped_timestamps = timestamps[::16]                  
        fs=len(reshaped_timestamps)/duration

        print("emg_loaded reshaped: ", raw_emg.shape)
        print("timestamps reshaped: ", len(reshaped_timestamps))
        print(f'fs reshaped: {fs:.2f} Hz\n')



        print("Reshaping GapWatch data completed single.\n")
        print("Loading GapWatch data completed single.\n")
        raw_emg = np.array(raw_emg)
        
        

        # Set each dataset starting and final value to zero for offset consistencies across concatenations
        
        # Normalize to zero the first value of the data to append
        raw_emg_normalized = raw_emg - raw_emg[:, 0:1]  # Set first value to 0 # Normalize each channel to have the first value as 0

        
        emg_data_combined = np.hstack((emg_data_combined, raw_emg_normalized))


        # Normalize to zero the last value of the data of origin        
        emg_data_combined = emg_data_combined - emg_data_combined[:, -1:]  # Set last value to 0



        timestamps_combined.extend(reshaped_timestamps)
        duration_combined += duration
        total_ts_reshaped += len(reshaped_timestamps)
        

    print("\nCOMBINED LOADING-RESHAPING COMPLETE.\n")


    # Timestamps reshaping & frequency calculation
    fs=total_ts_reshaped/duration_combined

    # Print shape information of calibration data
    print("\nInsights into final GapWatch data:")
    print(f" - Reshaped EMG shape: {emg_data_combined.shape}")        # Should be (n_samples, n_channels)
    print(f" - EMG Timestamps count: {total_ts_reshaped}")# Should be (n_samples)
    print(f" - Duration of EMG data: {duration_combined:.2f} s")
    print(f" - Sampling frequency fs of EMG data : {fs:.2f} Hz\n")

    return emg_data_combined, total_ts_reshaped, fs



#-------------------------------------------------------------------------------------
from config import *
from utils.utils_loading import *



def load_combined_reshape_vicon(selected_paths, topic_emg='/emg'):
    
    emg_data_combined = np.empty((16, 0))
    timestamps_combined = []
    duration_combined = 0
    total_ts_reshaped = 0  

    pos_hand_final_combined = np.empty((3, 0))
    rot_hand_final_combined = np.empty((4, 0))
    pos_f1_final_combined =   np.empty((3, 0))
    pos_f2_final_combined =   np.empty((3, 0))
    pos_f3_final_combined =   np.empty((3, 0))
    pos_f4_final_combined =   np.empty((3, 0))
    pos_f5_final_combined =   np.empty((3, 0))
    
    for bag_path in selected_paths:
        
        bag_path_emg = 'dataset/bag_emg' + bag_path
        bag_path_vicon = 'dataset/bag_vicon' + bag_path


        emg_data = []
        timestamps = []
        

        print("\nLoading GapWatch data single...")
        # Open the bag and extract EMG values from messages
        with rosbag.Bag(bag_path_emg, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_emg]):
                try:
                    for i in msg.emg:  # Read each value in the EMG array
                        emg_data.append(i)
                        timestamps.append(t.to_sec())
                except AttributeError as e:
                    print("Message missing expected fields:", e)
                    break

        emg_data = np.array(emg_data)
        timestamps = np.array(timestamps)
        print("emg_loaded raw: ", emg_data.shape)
        print("timestamps raw: ", timestamps.shape[0])
        duration = timestamps[-1] -timestamps[0]
        print(f"duration: {duration:.2f} s\n")


        print("\nReshaping GapWatch data single...")
        # Signal reshaping
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

        reshaped_timestamps = timestamps[::16]                  
        fs=len(reshaped_timestamps)/duration

        print("emg_loaded reshaped: ", raw_emg.shape)
        print("timestamps reshaped: ", len(reshaped_timestamps))
        print(f'fs reshaped: {fs:.2f} Hz\n')



        print("Reshaping GapWatch data completed single.\n")
        print("Loading GapWatch data completed single.\n")
        raw_emg = np.array(raw_emg)
        

        # Set each dataset starting and final value to zero for offset consistencies across concatenations
        # Normalize to zero the first value of the data to append
        raw_emg_normalized = raw_emg - raw_emg[:, 0:1]  # Set first value to 0 # Normalize each channel to have the first value as 0
        emg_data_combined = np.hstack((emg_data_combined, raw_emg_normalized))
        # Normalize to zero the last value of the data of origin        
        emg_data_combined = emg_data_combined - emg_data_combined[:, -1:]  # Set last value to 0

        timestamps_combined.extend(reshaped_timestamps)
        duration_combined += duration
        total_ts_reshaped += len(reshaped_timestamps)
        



        hand_positions, hand_orientations, marker_positions, timestamp_vicon = load_vicon_data(bag_path_vicon, topic_hand='/tf', topic_marker='/vicon/unlabeled_markers')
        pos_hand = hand_positions  
        rot_hand = hand_orientations
        pos_f1 = marker_positions[0]
        pos_f2 = marker_positions[1]
        pos_f3 = marker_positions[2]
        pos_f4 = marker_positions[3]
        pos_f5 = marker_positions[4]
        min_length = min(len(pos_hand), len(rot_hand), len(pos_f1), len(pos_f2), len(pos_f3), len(pos_f4), len(pos_f5))

        n_of_times = round(len(reshaped_timestamps)/min_length)
        print("Number of times to repeat Vicon data to match Gapwatch data:", n_of_times)

        pos_hand_final = []
        rot_hand_final = []
        pos_f1_final = []
        pos_f2_final = []
        pos_f3_final = []
        pos_f4_final = []
        pos_f5_final = []


        for i in range(min_length):
            for j in range(n_of_times):                
                pos_hand_final.append(pos_hand[i])
                rot_hand_final.append(rot_hand[i])
                pos_f1_final.append(pos_f1[i])
                pos_f2_final.append(pos_f2[i])
                pos_f3_final.append(pos_f3[i])
                pos_f4_final.append(pos_f4[i])
                pos_f5_final.append(pos_f5[i])
        print("hand model ", len(pos_hand_final))
        # Ensure the sigma list has the same number of elements as the EMG data
        sigma_len = len(pos_hand_final)
        if sigma_len < len(reshaped_timestamps):
            for k in range(len(reshaped_timestamps) - sigma_len):
                pos_hand_final.append(pos_hand[-1])
                rot_hand_final.append(rot_hand[-1])
                pos_f1_final.append(pos_f1[-1])
                pos_f2_final.append(pos_f2[-1])
                pos_f3_final.append(pos_f3[-1])
                pos_f4_final.append(pos_f4[-1])
                pos_f5_final.append(pos_f5[-1])

        if sigma_len > len(reshaped_timestamps):
            pos_hand_final = pos_hand_final[:len(reshaped_timestamps)]
            rot_hand_final = rot_hand_final[:len(reshaped_timestamps)]
            pos_f1_final = pos_f1_final[:len(reshaped_timestamps)]
            pos_f2_final = pos_f2_final[:len(reshaped_timestamps)]
            pos_f3_final = pos_f3_final[:len(reshaped_timestamps)]
            pos_f4_final = pos_f4_final[:len(reshaped_timestamps)]
            pos_f5_final = pos_f5_final[:len(reshaped_timestamps)]
        
        print(pos_hand_final[-1])
        # Reaassign after modifications
        sigma_mot_len = len(pos_hand_final)


        print(f" - Sigma Motion samples single: {sigma_mot_len}")
        print(f" - EMG data samples single: {len(reshaped_timestamps)}\n")



        
        pos_hand_final = np.array(pos_hand_final).T
        rot_hand_final = np.array(rot_hand_final).T
        pos_f1_final = np.array(pos_f1_final).T
        pos_f2_final = np.array(pos_f2_final).T
        pos_f3_final = np.array(pos_f3_final).T
        pos_f4_final = np.array(pos_f4_final).T
        pos_f5_final = np.array(pos_f5_final).T
        print("pos_hand", pos_hand_final.shape)

        pos_hand_final_combined = np.hstack((pos_hand_final_combined, pos_hand_final))
        rot_hand_final_combined = np.hstack((rot_hand_final_combined, rot_hand_final))
        pos_f1_final_combined =   np.hstack((pos_f1_final_combined, pos_f1_final))
        pos_f2_final_combined =   np.hstack((pos_f2_final_combined, pos_f2_final))
        pos_f3_final_combined =   np.hstack((pos_f3_final_combined, pos_f3_final))
        pos_f4_final_combined =   np.hstack((pos_f4_final_combined, pos_f4_final))
        pos_f5_final_combined =   np.hstack((pos_f5_final_combined, pos_f5_final))
        print("pos_hand_final", pos_hand_final_combined.shape)

    print("\nCOMBINED LOADING-RESHAPING COMPLETE.\n")




    # Timestamps reshaping & frequency calculation
    fs=total_ts_reshaped/duration_combined

    # Print shape information of calibration data
    print("\nInsights into final GapWatch data:")
    print(f" - Reshaped EMG shape: {emg_data_combined.shape}")        # Should be (n_samples, n_channels)
    print(f" - EMG Timestamps count: {total_ts_reshaped}")# Should be (n_samples)
    print(f" - Duration of EMG data: {duration_combined:.2f} s")
    print(f" - Sampling frequency fs of EMG data : {fs:.2f} Hz\n")

    return emg_data_combined, total_ts_reshaped, fs, pos_hand_final_combined, rot_hand_final_combined, pos_f1_final_combined, pos_f2_final_combined, pos_f3_final_combined, pos_f4_final_combined, pos_f5_final_combined