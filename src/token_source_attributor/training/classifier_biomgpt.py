import torch
from torch.utils.data import DataLoader, random_split

from token_source_attributor.data.dataset import BinnedMicrobiomeClassificationDataset
from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForSequenceClassification,
)

from pathlib import Path
CHECKPOINT_DIR = Path("checkpoints_classifier")


def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BinnedMicrobiomeClassificationDataset(
        path="src/token_source_attributor/data/classifier_stool_binned.tsv",
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
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

    # fine tuning the backbone with a classification head on top, initialized with pretrained weights from MLM task
    backbone_state_dict = torch.load(
        "checkpoint/biomgpt_mlm_best_backbone.pt",
        map_location=device,
    )

    backbone.load_state_dict(backbone_state_dict)

    model = BioMGPTForSequenceClassification(backbone, num_classes=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(30):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            species_ids = batch["species_ids"].to(device)
            abundance_bins = batch["abundance_bins"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(
                species_ids=species_ids,
                abundance_bins=abundance_bins,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = out["loss"]

            # zero out the gradient ∇ from previous batch
            optimizer.zero_grad()
            # calculate the gradient ∇... [∂L/∂w_1, ∂L/∂w_2, ... , ∂L/∂w_T]
            loss.backward()
            # w = w - lr * ∇
            optimizer.step()

            total_train_loss += loss.item()

        # log the average loss for training epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                species_ids = batch["species_ids"].to(device)
                abundance_bins = batch["abundance_bins"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                out = model(
                    species_ids=species_ids,
                    abundance_bins=abundance_bins,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = out["loss"]
                logits = out["logits"]

                total_val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss = {avg_train_loss:.4f}, "
            f"val_loss = {avg_val_loss:.4f}, "
            f"val_acc = {val_acc:.4f}"
        )

        # save last checkpoint
        # torch.save(
        #     model.state_dict(),
        #     CHECKPOINT_DIR / "biomgpt_classify_last.pt",
        # )


         # save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "biomgpt_classify_best.pt",
            )
            print("Saved new best checkpoint ✅")


if __name__ == "__main__":
    train()