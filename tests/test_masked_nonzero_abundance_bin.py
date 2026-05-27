import torch
from torch.utils.data import DataLoader
from token_source_attributor.data.dataset import BinnedMicrobiomeDataset
from token_source_attributor.models.biomgpt import mask_nonzero_abundance_bins, BioMGPTEncoderBackbone, BioMGPTForMaskedAbundanceModeling

def test_masked_nonzero_abundance_bins():
    MASKED_TOKEN = 51
    COVERAGE = 0.25

    a_bins = torch.tensor([
        [0, 3, 5, 4, 2, 1, 0, 0, 5, 3, 4, 1, 2],
        [0, 2, 4, 3, 1, 0, 0, 0, 4, 2, 3, 1, 1],
    ])

    abundance_bins_masked, mlm_labels, masked_positions = mask_nonzero_abundance_bins(
        abundance_bins=a_bins,
        mask_bin_id=MASKED_TOKEN,
        mask_prob=COVERAGE,
    )

    masked_positions = abundance_bins_masked == MASKED_TOKEN
    labeled_positions = mlm_labels != -100
    assert torch.equal(masked_positions, labeled_positions)
    
    #that all labeled positions are for nonzero values 
    assert ((labeled_positions != -100) & (labeled_positions == 0)).any()
    
    
    mask_count = (mlm_labels != -100).sum().item()
    element_count = a_bins.numel()
    print('masked count',mask_count)
    print('element count', element_count)
    print()
    print('desired coverage: ', COVERAGE)
    print('masked coverage: ', mask_count / element_count)
    print()
    print("Abundance bins masked:", abundance_bins_masked)
    print("MLM labels:", mlm_labels)
    print("Masked positions:", masked_positions)
    
    print("Test passed! ✅")

if __name__ == "__main__":
    test_masked_nonzero_abundance_bins()