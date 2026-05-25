import numpy as np  
from token_source_attributor.data.data_preprocessing import bin_microbiome_sample

def test_binning():
    abundance_vector = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
    binned_vector = bin_microbiome_sample(abundance_vector, B=5)
    assert len(binned_vector) == len(abundance_vector)
    assert all(0 <= x < 5 for x in binned_vector)

    print("Binned vector:", binned_vector)
    print("Test passed! ✅")

if __name__ == "__main__":
    test_binning()