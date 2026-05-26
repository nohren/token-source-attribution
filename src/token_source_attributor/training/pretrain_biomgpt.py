# pretrain masked abundance modeling head and backbone
import torch
from torch.utils.data import DataLoader

from token_source_attributor.data.dataset import BinnedMicrobiomeDataset
from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForMaskedAbundanceModeling,
    mask_nonzero_abundance_bins,
)


def train():
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    model.train()
    print("Starting training...")
    for epoch in range(30):
        total_loss = 0.0

        for batch in dataloader:
            species_ids = batch["species_ids"].to(device)
            abundance_bins = batch["abundance_bins"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            abundance_bins_masked, mlm_labels, masked_positions = mask_nonzero_abundance_bins(
                abundance_bins=abundance_bins,
                mask_bin_id=model.backbone.mask_bin_id,
                mask_prob=0.25,
            )

            out = model(
                species_ids=species_ids,
                abundance_bins_masked=abundance_bins_masked,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
            )

            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            f"checkpoints/biomgpt_pretrain_epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    train()