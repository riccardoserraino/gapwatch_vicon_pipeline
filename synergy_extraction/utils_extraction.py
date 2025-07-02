from config import *


####################################################################################################
# Synergies extraction and reconstruction fucntions
####################################################################################################



####################################################################################################
# PCA
####################################################################################################

def pca_emg(emg_data, n_components, scale_W=False, random_state=None, svd_solver='full'):
    """
    Applies Principal Component Analysis (PCA) to EMG data for dimensionality reduction 
    and synergy extraction.

    Args:
        emg_data (ndarray): Input EMG data matrix of shape (n_samples, n_muscles).
        n_components (int): Number of principal components (synergies) to extract.
        scale_W (bool): If True, scale scores (U) by the explained variance ratio.
        random_state (int or None): Random seed for reproducibility.
        svd_solver (str): SVD solver to use. Default is 'full' (LAPACK-based).

    Returns:
        H (ndarray): Principal components (muscle synergies), shape (n_components, n_muscles).
        W (ndarray): Projection of data onto components (temporal activations), shape (n_samples, n_components).
        mean (ndarray): Mean of the original data used for reconstruction.
        X_reconstructed (ndarray): Reconstructed EMG data using the selected principal components.
    """

    print("\nApplying PCA...")

    #preprocessing data for a clear reconstruction (centering and normalization)
    X_centered = emg_data - np.mean(emg_data, axis=0)
    X_normalized = (emg_data - np.mean(emg_data, axis=0)) / np.std(emg_data, axis=0)

    #model pca
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    
    #extracting matrices
    W = pca.fit_transform(emg_data)             # Synergies over time
    H = pca.components_                         # Muscle patterns (Muscular synergy matrix)
    mean = pca.mean_
    
    # Scale scores by explained variance (makes them more comparable)
    if scale_W:
        W = W * np.sqrt(pca.explained_variance_ratio_)
    # Transpose to keep same structure as NMF function
    if H.shape[0] != n_components:
        H = H.T     # Ensure S_m has shape (n_synergies, n_muscles)
    
    #reconstruction based on the inverse transform
    X_transformed = pca.fit_transform(X_centered) # Neural matrix (synergies over time) adjusted for centering wrt original data and enforce positive values
    X_reconstructed = pca.inverse_transform(X_transformed) + np.mean(emg_data, axis=0) # the mean is added to enforce values of synergies and reconstruction being non negative as the original data

    """mse = np.mean((emg_data - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse}")"""

    print("PCA completed.\n")

    return H, W, mean, X_reconstructed


#---------------------------------------------------------------------------------------------
def pca_emg_reconstruction(W, H, mean, n_synergies):
    """
    Reconstructs EMG data using a selected number of PCA components.

    Args:
        W (ndarray): Scores matrix (temporal activations), shape (n_samples, total_components).
        H (ndarray): Principal components (muscle synergies), shape (total_components, n_muscles).
        mean (ndarray): Mean vector used for centering during PCA.
        n_synergies (int): Number of components to use for reconstruction.

    Returns:
        reconstructed (ndarray): Reconstructed EMG data matrix, shape (n_samples, n_muscles).
    """

    print(f"\nReconstructing the signal with {n_synergies} synergies...")
    
    # Select the first n_components
    W_rec = W[:, :n_synergies]
    H_rec = H[:n_synergies, :]
    
    # Reconstruct the data
    reconstructed = np.dot(W_rec, H_rec) + mean

    print("Reconstruction completed.\n")

    return reconstructed



####################################################################################################
# NMF functions
####################################################################################################

#---------------------------------------------------------------------------------------------
def nmf_emg(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    """
    Applies Non-negative Matrix Factorization (NMF) to extract muscle synergies 
    and their activations from EMG data.

    Args:
        emg_data (ndarray): Input EMG data matrix of shape (n_muscle, n_samples).
        n_components (int): Number of synergies (components) to extract.
        init (str): Initialization method for NMF (e.g. for sparse, 'nndsvd', 'random').
        max_iter (int): Maximum number of iterations before stopping.
        l1_ratio (float): L1 regularization ratio (between 0 and 1).
        alpha_W (float): Regularization strength for the activation matrix U (sparsity coefficient).
        random_state (int): Random seed for reproducibility.

    Returns:
        W (ndarray): Muscle synergy matrix (muscle weights), shape (n_muscles, n_components).
        H (ndarray): Synergy activations over time (neural drive), shape (n_components, n_samples).
    """

    print("\nApplying NMF...")
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    H_T = nmf.fit_transform(emg_data)         # Synergy activations over time (Neural drive matrix)
    W_T = nmf.components_                     # Channels patterns (Muscular synergy activation matrix)
 
 
    # Transpose W and H to match the correct shapes if needed
    if H_T.shape[0] == emg_data.shape[0]:
        H = H_T.T         
    else:
        H = H_T
    if W_T.shape[0] == n_components:
        W = W_T.T 
    else:
        W = W_T   

    print("NMF completed.\n")

    return W, H



#---------------------------------------------------------------------------------------------
def nmf_emg_reconstruction(W, H, final_emg_for_nmf):
    """
    Reconstructs the EMG signal using a selected number of NMF components (synergies).
    
    Args:
        W (ndarray): Muscle synergy matrix (muscle weights), shape (n_muscles, n_components).
        H (ndarray): Synergy activations over time (neural drive), shape (n_components, n_samples).
        final_emg_for_nmf (ndarray): Original EMG data matrix used for NMF, shape (n_samples, n_muscles).
    Returns:
        reconstructed (ndarray): Reconstructed EMG data, shape (n_muscles, n_samples).
    """

    print(f"\nReconstructing the signal with selected number of synergies...")
    reconstructed_nmf = np.dot(W, H)
    # Ensure the reconstructed signal has the same shape as the original EMG data fro plotting comparisons
    if reconstructed_nmf.shape[0] != final_emg_for_nmf.shape[0]:
        reconstructed_nmf = reconstructed_nmf.T  # Transpose if needed to match (n_channels, n_samples)
    print("Reconstruction check completed.\n")

    return reconstructed_nmf



####################################################################################################
# Reconstruction error functions
####################################################################################################

def rmse(X, X_estimated):

    """
    Computes the Root Mean Square Error (RMSE) between original and estimated data.
    Used to compare the original signal (filtered) with the reconstruction based on the number of synergies extracted (through dot product).
    It is recommended to use it to evaluate the accuracy in reconstructing the original signal based on different number of synergies selected, 
    the best option to visualize the results is with a histogram where we compare different extraction-reconstruction methods.

    Args:
        X: Original data matrix (n_samples x n_features) 
        X_estimated: Estimated data matrix (n_samples x n_features)

    Outputs:
        rmse_value: RMSE value (accuracy)
    """

    X = np.array(X)
    X_estimated = np.array(X_estimated)

    if X.shape != X_estimated.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    mse = np.mean((X-X_estimated)**2)

    rmse = np.sqrt(mse)

    return 1 - rmse





