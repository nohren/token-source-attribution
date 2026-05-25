import numpy as np


#TODO what is B hyperparameter in discretization? scBERT says 7 in their github or num tokens
def bin_microbiome_sample(abundance_vector, B=7):
    """
    Bins a single continuous 1D array of species abundances.
    abundance_vector: shape (num_species,)
    """
    # Initialize output array with zeros (Bin 0)
    binned_vector = np.zeros_like(abundance_vector, dtype=int)
    
    # Get indices of non-zero features
    non_zero_indices = np.where(abundance_vector > 0)[0]
    N = len(non_zero_indices)
    
    if N == 0:
        return binned_vector
        
    # Sort the non-zero indices by their raw abundance values descending
    sorted_nz_indices = non_zero_indices[np.argsort(-abundance_vector[non_zero_indices])]
    
    if N >= B:
        # Standard even splitting into B chunks
        chunks = np.array_split(sorted_nz_indices, B)
        # array_split outputs from first to last, so chunks[0] is the highest abundance
        for bin_idx, chunk in enumerate(chunks):
            actual_bin_value = B - bin_idx  # Maps chunk 0 -> Bin B, chunk B-1 -> Bin 1
            binned_vector[chunk] = actual_bin_value
    else:
        # Low diversity override: scale proportionally across 1 to B
        # Rank values scaled from 1 to B
        proportional_bins = np.linspace(B, 1, num=N).astype(int)
        for idx, actual_bin_value in zip(sorted_nz_indices, proportional_bins):
            binned_vector[idx] = actual_bin_value
            
    return binned_vector