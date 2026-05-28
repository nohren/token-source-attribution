import torch
from torch.utils.data import DataLoader

from token_source_attributor.data.dataset import BinnedMicrobiomeClassificationDataset, get_species_name
from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForSequenceClassification,
)
from token_source_attributor.attribution.saliency import saliency_ibd_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = BinnedMicrobiomeClassificationDataset(
    path="src/token_source_attributor/data/classifier_stool_binned.tsv",
)

# load data
loader = DataLoader(dataset, batch_size=32, shuffle=True)

backbone = BioMGPTEncoderBackbone(
    num_species=dataset.num_species,
    max_bin=50,
    d_model=512,
    n_layers=8,
    n_heads=8,
    dim_feedforward=512,
    dropout=0.1,
)

model = BioMGPTForSequenceClassification(
    backbone=backbone,
    num_classes=2,
).to(device)

model.load_state_dict(
    torch.load("checkpoints_classifier/biomgpt_classify_1st_run_val-loss-0.21_val-acc-0.93acc.pt", map_location=device)
)

# just use the first batch
batch = next(iter(loader))

result = saliency_ibd_batch(
    model=model,
    batch=batch,
    device=device,
    target_class=1,  # IBD logit
)

species_attr = result["species_attr"]       # [B, S]
abundance_attr = result["abundance_attr"]   # [B, S]
total_attr = result["total_attr"]           # [B, S]

# generate top k attr per sample in batch [B,S]
top_k = 5 
values, indices = total_attr.topk(top_k, dim=1)

# iterate over all samples in the batch
for b in range(indices.size(0)):
    print("=" * 80)
    print("sample_id:", batch["sample_id"][b])
    print("study_id:", batch["study_id"][b])
    print("disease:", batch["disease"][b])
    print("label:", batch["label"][b].item())
    print("pred:", result["preds"][b].item())
    print("prob_healthy:", result["probs"][b, 0].item())
    print("prob_ibd:", result["probs"][b, 1].item())
    print("confidence:", result["confidence"][b].item())
    print("entropy:", result["entropy"][b].item())
    
    
    print("\nTop species by saliency toward IBD logit:")
    # for this sample in this batch list out the top 5 species by saliency
    for rank in range(top_k):
        species_idx = indices[b, rank].item()
        species_clade_name = dataset.species_cols[species_idx]
        bin_value = batch["abundance_bins"][b, species_idx].item()

        print(
            rank + 1,
            get_species_name(species_clade_name),
            "bin=", bin_value,
            "\nspecies_attr=", species_attr[b, species_idx].item(),
            "abundance_attr=", abundance_attr[b, species_idx].item(),
            "\ntotal_attr=", total_attr[b, species_idx].item(),
        )