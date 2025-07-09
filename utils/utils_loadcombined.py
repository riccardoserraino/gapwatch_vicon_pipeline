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
    start_cut = 300
    end_cut = 300   


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
        zero_column = np.zeros((16, 100))

        
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



