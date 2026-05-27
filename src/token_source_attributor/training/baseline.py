import torch
from torch.utils.data import DataLoader

from token_source_attributor.data.dataset import BinnedMicrobiomeDataset
from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForMaskedAbundanceModeling,
    mask_nonzero_abundance_bins,
)
# predict the mean as the masked abundance bin value 
# what is MSE?

def baseline():
   
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BinnedMicrobiomeDataset(
        path="src/token_source_attributor/data/bert_pretraining_stool_binned.tsv",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )
    num_species = dataset.num_species
    
    backbone = BioMGPTEncoderBackbone(
        num_species=num_species,
        max_bin=50,
        d_model=512,
        n_layers=8,
        n_heads=8,
        dim_feedforward=512,
        dropout=0.1,
    )
    model = BioMGPTForMaskedAbundanceModeling(backbone).to(device)

    all_targets = []

    for batch in dataloader:
        abundance_bins = batch["abundance_bins"].to(device)

        abundance_bins_masked, mlm_labels, _ = mask_nonzero_abundance_bins(
            abundance_bins=abundance_bins,
            mask_bin_id=model.backbone.mask_bin_id,
            mask_prob=0.25,
        )

        mask = mlm_labels != -100
        all_targets.append(mlm_labels[mask].float().cpu())

    all_targets = torch.cat(all_targets)
    mean_target = all_targets.mean()

    # predict the mean baseline mse
    baseline_mse = ((all_targets - mean_target) ** 2).mean()
    baseline_rmse = baseline_mse.sqrt()

    print("mean target:", mean_target.item())
    print("baseline MSE:", baseline_mse.item())
    print("baseline RMSE:", baseline_rmse.item())

if __name__ == "__main__":
    baseline()

