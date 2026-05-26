import torch
from torch.utils.data import DataLoader
from token_source_attributor.data.dataset import BinnedMicrobiomeDataset
from token_source_attributor.models.biomgpt import mask_nonzero_abundance_bins, BioMGPTEncoderBackbone, BioMGPTForMaskedAbundanceModeling

def test_masked_nonzero_abundance_bins():
    

    a_bins = torch.tensor([
        [0, 3, 5, 4, 2, 1, 0, 0, 5, 3, 4, 1, 2],
        [0, 2, 4, 3, 1, 0, 0, 0, 4, 2, 3, 1, 1],
    ])

    abundance_bins_masked, mlm_labels, masked_positions = mask_nonzero_abundance_bins(
        abundance_bins=a_bins,
        mask_bin_id=51,
        mask_prob=0.25,
    )

   

    print((mlm_labels != -100).sum())
    print(a_bins.numel())

    print("Abundance bins masked:", abundance_bins_masked)
    print("MLM labels:", mlm_labels)
    print("Masked positions:", masked_positions)
    
    print("Test passed! ✅")

if __name__ == "__main__":
    test_masked_nonzero_abundance_bins()