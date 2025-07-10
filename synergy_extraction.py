from config import *
# Utils functions
from utils.utils_loading import *
from utils.utils_motion import * 
from utils.utils_synergies import *
from utils.utils_visual import *
from utils.utils_loadcombined import *




########################################################################
# Initialization - Set ROS topic for Gapwatch and Vicon and set bagfile paths
########################################################################

# set topic
emg_topic = '/emg'

# Dictionary mapping numbers to dataset paths
dataset_options_emg = {
    '1':   'dataset/bag_emg/power_grasp1.bag',
    '2':   'dataset/bag_emg/power_grasp2.bag',
    '3':   'dataset/bag_emg/power_grasp3.bag',
    '4':   'dataset/bag_emg/power_grasp4.bag',
    '5':   'dataset/bag_emg/power_grasp5.bag',
    '6':   'dataset/bag_emg/power_grasp6.bag',
    '7':   'dataset/bag_emg/power_grasp7.bag',
    '8':   'dataset/bag_emg/power_grasp8.bag',
    '9':   'dataset/bag_emg/power_grasp9.bag',
    '10':  'dataset/bag_emg/power_grasp10.bag',
    '11':  'dataset/bag_emg/pinch1.bag',
    '12':  'dataset/bag_emg/pinch2.bag',
    '13':  'dataset/bag_emg/pinch3.bag',
    '14':  'dataset/bag_emg/pinch4.bag',
    '15':  'dataset/bag_emg/pinch5.bag',
    '16':  'dataset/bag_emg/pinch6.bag',
    '17':  'dataset/bag_emg/pinch7.bag',
    '18':  'dataset/bag_emg/pinch8.bag',
    '19':  'dataset/bag_emg/pinch9.bag',
    '20':  'dataset/bag_emg/pinch10.bag',
    '21':  'dataset/bag_emg/ulnar1.bag',
    '22':  'dataset/bag_emg/ulnar2.bag',
    '23':  'dataset/bag_emg/ulnar3.bag',
    '24':  'dataset/bag_emg/ulnar4.bag',
    '25':  'dataset/bag_emg/ulnar5.bag',
    '26':  'dataset/bag_emg/ulnar6.bag',
    '27':  'dataset/bag_emg/ulnar7.bag',
    '28':  'dataset/bag_emg/ulnar8.bag',
    '29':  'dataset/bag_emg/ulnar9.bag',
    '30':  'dataset/bag_emg/ulnar10.bag',
    '31':  'dataset/bag_emg/sto1.bag',
    '32':  'dataset/bag_emg/sto2.bag',
    '33':  'dataset/bag_emg/sto3.bag',
    '34':  'dataset/bag_emg/sto4.bag',
    '35':  'dataset/bag_emg/sto5.bag',
    '36':  'dataset/bag_emg/sto6.bag',
    '37':  'dataset/bag_emg/sto7.bag',
    '38':  'dataset/bag_emg/sto8.bag',
    '39':  'dataset/bag_emg/sto9.bag',
    '40':  'dataset/bag_emg/sto10.bag',
    '41':  'dataset/bag_emg/thumb_up1.bag',
    '42':  'dataset/bag_emg/thumb_up2.bag',
    '43':  'dataset/bag_emg/thumb_up3.bag',
    '44':  'dataset/bag_emg/thumb_up4.bag',
    '45':  'dataset/bag_emg/thumb_up5.bag',
    '46':  'dataset/bag_emg/thumb_up6.bag',
    '47':  'dataset/bag_emg/thumb_up7.bag',
    '48':  'dataset/bag_emg/thumb_up8.bag',
    '49':  'dataset/bag_emg/thumb_up9.bag',
    '50':  'dataset/bag_emg/thumb_up10.bag',
    '51':  'dataset/bag_emg/thumb1.bag',
    '52':  'dataset/bag_emg/thumb2.bag',
    '53':  'dataset/bag_emg/thumb3.bag',
    '54':  'dataset/bag_emg/thumb4.bag',
    '55':  'dataset/bag_emg/thumb5.bag',
    '56':  'dataset/bag_emg/thumb6.bag',
    '57':  'dataset/bag_emg/thumb7.bag',
    '58':  'dataset/bag_emg/thumb8.bag',
    '59':  'dataset/bag_emg/thumb9.bag',
    '60':  'dataset/bag_emg/thumb10.bag',
    '61':  'dataset/bag_emg/index1.bag',
    '62':  'dataset/bag_emg/index2.bag',
    '63':  'dataset/bag_emg/index3.bag',
    '64':  'dataset/bag_emg/index4.bag',
    '65':  'dataset/bag_emg/index5.bag',
    '66':  'dataset/bag_emg/index6.bag',
    '67':  'dataset/bag_emg/index7.bag',
    '68':  'dataset/bag_emg/index8.bag',
    '69':  'dataset/bag_emg/index9.bag',
    '70':  'dataset/bag_emg/index10.bag',
    '71':  'dataset/bag_emg/middle1.bag',
    '72':  'dataset/bag_emg/middle2.bag',
    '73':  'dataset/bag_emg/middle3.bag',
    '74':  'dataset/bag_emg/middle4.bag',
    '75':  'dataset/bag_emg/middle5.bag',
    '76':  'dataset/bag_emg/middle6.bag',
    '77':  'dataset/bag_emg/middle7.bag',
    '78':  'dataset/bag_emg/middle8.bag',
    '79':  'dataset/bag_emg/middle9.bag',
    '80':  'dataset/bag_emg/middle10.bag',
    '81':  'dataset/bag_emg/ring1.bag',
    '82':  'dataset/bag_emg/ring2.bag',
    '83':  'dataset/bag_emg/ring3.bag',
    '84':  'dataset/bag_emg/ring4.bag',
    '85':  'dataset/bag_emg/ring5.bag',
    '86':  'dataset/bag_emg/ring6.bag',
    '87':  'dataset/bag_emg/ring7.bag',
    '88':  'dataset/bag_emg/ring8.bag',
    '89':  'dataset/bag_emg/ring9.bag',
    '90':  'dataset/bag_emg/ring10.bag',
    '91':  'dataset/bag_emg/little1.bag',
    '92':  'dataset/bag_emg/little2.bag',
    '93':  'dataset/bag_emg/little3.bag',
    '94':  'dataset/bag_emg/little4.bag',
    '95':  'dataset/bag_emg/little5.bag',
    '96':  'dataset/bag_emg/little6.bag',
    '97':  'dataset/bag_emg/little7.bag',
    '98':  'dataset/bag_emg/little8.bag',
    '99':  'dataset/bag_emg/little9.bag',
    '100': 'dataset/bag_emg/little10.bag',
    '101': 'dataset/bag_emg/bottle1.bag',
    '102': 'dataset/bag_emg/bottle2.bag',
    '103': 'dataset/bag_emg/bottle3.bag',
    '104': 'dataset/bag_emg/bottle4.bag',
    '105': 'dataset/bag_emg/bottle5.bag',
    '106': 'dataset/bag_emg/bottle6.bag',
    '107': 'dataset/bag_emg/bottle7.bag',
    '108': 'dataset/bag_emg/bottle8.bag',
    '109': 'dataset/bag_emg/bottle9.bag',
    '110': 'dataset/bag_emg/bottle10.bag',
    '111': 'dataset/bag_emg/pen1.bag',
    '112': 'dataset/bag_emg/pen2.bag',
    '113': 'dataset/bag_emg/pen3.bag',
    '114': 'dataset/bag_emg/pen4.bag',
    '115': 'dataset/bag_emg/pen5.bag',
    '116': 'dataset/bag_emg/pen6.bag',
    '117': 'dataset/bag_emg/pen7.bag',
    '118': 'dataset/bag_emg/pen8.bag',
    '119': 'dataset/bag_emg/pen9.bag',
    '120': 'dataset/bag_emg/pen10.bag',
    '121': 'dataset/bag_emg/tablet1.bag',
    '122': 'dataset/bag_emg/tablet2.bag',
    '123': 'dataset/bag_emg/tablet3.bag',
    '124': 'dataset/bag_emg/tablet4.bag',
    '125': 'dataset/bag_emg/tablet5.bag',
    '126': 'dataset/bag_emg/tablet6.bag',
    '127': 'dataset/bag_emg/tablet7.bag',
    '128': 'dataset/bag_emg/tablet8.bag',
    '129': 'dataset/bag_emg/tablet9.bag',
    '130': 'dataset/bag_emg/tablet10.bag',
    '131': 'dataset/bag_emg/pinza1.bag',
    '132': 'dataset/bag_emg/pinza2.bag',
    '133': 'dataset/bag_emg/pinza3.bag',
    '134': 'dataset/bag_emg/pinza4.bag',
    '135': 'dataset/bag_emg/pinza5.bag',
    '136': 'dataset/bag_emg/pinza6.bag',
    '137': 'dataset/bag_emg/pinza7.bag',
    '138': 'dataset/bag_emg/pinza8.bag',
    '139': 'dataset/bag_emg/pinza9.bag',
    '140': 'dataset/bag_emg/pinza10.bag',
    '141': 'dataset/bag_emg/phone1.bag',
    '142': 'dataset/bag_emg/phone2.bag',
    '143': 'dataset/bag_emg/phone3.bag',
    '144': 'dataset/bag_emg/phone4.bag',
    '145': 'dataset/bag_emg/phone5.bag',
    '146': 'dataset/bag_emg/phone6.bag',
    '147': 'dataset/bag_emg/phone7.bag',
    '148': 'dataset/bag_emg/phone8.bag',
    '149': 'dataset/bag_emg/phone9.bag',
    '150': 'dataset/bag_emg/phone10.bag'
}

#Set base paths
base_path = {
    "bag_emg": "dataset/bag_emg/"
}

# Set train dataset as a constant
train_dataset = "power_grasp1.bag"     # <-- Change here to use a different file

# Ask user to select datasets and their order
selected_paths = select_datasets(dataset_options_emg)



########################################################################
# Data Loading - Read Gapwatch, Vicon data from selected ROS bag files
########################################################################

#-----------------------------------------------------------------------
# Load EMG train data from a train bag file 
#-----------------------------------------------------------------------

bag_path_emg = base_path['bag_emg'] + train_dataset     

emg_data_train, timestamps_train = load_emg_data(bag_path_emg, emg_topic)

# Data Reshaping 1 - Reshape raw EMG vector into (16 x N) matrix format
final_emg_train, timestamps_emg_train, fs_emg_train = reshape_emg_data(emg_data_train, timestamps_train)
#aligned_emg_train = np.array(align_signal_baselines(final_emg_train, method='mean'))
#plot_emg(aligned_emg_train, title='Raw EMG Signals - Mean Value Aligned')
#plot_emg_channels_2cols(final_emg_train)


# Data Filtering 2 - Filter EMG data with EMG filtering specs
filtered_emg_train = np.array([preprocess_emg(final_emg_train[i, :], fs=fs_emg_train) for i in range(final_emg_train.shape[0])])
#plot_emg(filtered_emg_train, title='Filtered EMG Signals')
#plot_emg_channels_2cols(filtered_emg_train)


#-----------------------------------------------------------------------
# Load EMG filtered test data from a test bag file 
#-----------------------------------------------------------------------

final_emg_data_test_combined, final_timestamps_test_combined, fs_test_combined = load_combined_reshape(selected_paths, emg_topic)

#plot_emg(final_emg_data_test_combined, title='Filtered EMG Signals')


########################################################################
# Gapwatch Data Processing - Extract synergies from the EMG train data 
#                          - Estimate the synergy activation patterns of the test data
#                          - Reconstruct the EMG test data from the extracted synergies
########################################################################

#------------------------------------------------------------------------
# Extract synergies from the EMG train data using Sparse NMF
#------------------------------------------------------------------------

optimal_synergies_nmf = 2

# Transpose for sklearn and plotting compatibility
final_emg_for_nmf = filtered_emg_train.T  

W, H = nmf_emg(final_emg_for_nmf, 
               n_components=optimal_synergies_nmf,
               init='nndsvd', 
               max_iter=500, 
               l1_ratio=0.15, 
               alpha_W=0.0005, 
               random_state=21)

# Plot original E, channel weights W, and activation over time H
plot_nmf(final_emg_for_nmf, W, H, optimal_synergies_nmf)


# Print shapes of extracted matrices
print("Insights into extracted NMF matrices:")
print(f" - Final EMG for NMF shape: {final_emg_for_nmf.shape}")   # Should be (n_samples, n_channels)
print(f" - Extracted Synergy Matrix W shape: {W.shape}")          # Should be (n_channels, n_synergies)
print(f" - Extracted Activation Matrix H shape: {H.shape}\n")     # Should be (n_synergies, n_samples)


# Pseudo inverse of H matrix (neural matrix representing activation patterns)
W_pinv = compute_pseudo_inverse(W) # Should be (n_synergies, n_channels)

print("\nEstimating new synergy matrix H using pseudo-inverse of W and test...")
# Estimate the synergy matrix H from the pseudo-inverse of W
estimated_H = np.dot(W_pinv, final_emg_data_test_combined)  
# Should be (n_synergies, testdata_n_samples) = (n_synergies, n_channels) X (n_channels, testdata_n_samples)
print("Estimation completed.\n")

# Print insights into the estimated synergy matrix
print("\nInsights into estimated synergy matrix:")
print(" - Pseudo-inverse of W shape:", W_pinv.shape)  # Should be (n_synergies, n_channels)
print(" - Filtered EMG test data shape:", final_emg_data_test_combined.shape)  # Should be (n_channels, testdata_n_samples)
print(f" - Estimated Synergy Matrix H from W_pinv shape: {estimated_H.shape} \n")   # Should be (n_synergies, testdata_n_samples)


# Reconstruct the EMG test data using the estimated synergy matrix H
print("\nReconstructing the EMG test data using estimated synergy matrix H...")
H_train = H
H_test = estimated_H
reconstructed_t = nmf_emg_reconstruction(W, H_test, final_emg_data_test_combined.T)
print(f" - Reconstructed EMG shape: {reconstructed_t.shape}\n") # Should be (testdata_n_samples, n_channels) after doing the transpose for plotting purposes
print("Reconstruction completed.\n")

plot_all_results(final_emg_data_test_combined.T, reconstructed_t, W, H_test, optimal_synergies_nmf)




########################################################################
# Sigma Matrix EMG - Compute the Sigma matrix for the EMG test data to define hand closure
########################################################################
# This approach has to be considered valid if the training dataset is the power_grasp1.bag
# Otherwise the approach needs changes to adapt to the new dataset to the train synegies weight patterns extracted

highest_value, correspondent_value, max_difference = find_max_difference(H_train)

sigma_emg = scale_differences(H_test, max_difference)
print("\nInsights into the flexion/extention synergy matrix:")
print(f" - Highest value in H_train row: {highest_value}")
print(f" - Corresponding value in H_train row+1: {correspondent_value}")
print(f" - Maximum difference in H_train: {max_difference}")
print(f" - Flexion/Extension synergy matrix shape: {sigma_emg.shape}\n")

plot_sigma_emg(sigma_emg, title='Sigma Matrix EMG')








