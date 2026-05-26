import numpy as np  
from token_source_attributor.data.data_preprocessing import bin_microbiome_sample

def test_binning():
    abundance_vector = np.array([0.0, 0.1, 0.5, 0.3, 0.05, 0.05, 0.0, 0.0, 0.5, 0.1, 0.3, 0.05, 0.1])
    binned_vector = bin_microbiome_sample(abundance_vector, B=5)
    print("Abundance vector:", abundance_vector)
    print("Binned vector:", binned_vector)
    assert len(binned_vector) == len(abundance_vector)
    assert set(binned_vector) <= set(range(6))  # Bins should be in {0, 1, 2, 3, 4, 5}
    assert np.array_equal(binned_vector, np.array([0, 3, 5, 4, 2, 1, 0, 0, 5, 3, 4, 1, 2]))
    print("Test passed! ✅")

if __name__ == "__main__":
    test_binning()