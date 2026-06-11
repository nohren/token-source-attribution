import torch

from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForSequenceClassification,
)


def test_cls_attention_matches_manual_average():
    torch.manual_seed(0)

    backbone = BioMGPTEncoderBackbone(
        num_species=6,
        max_bin=50,
        d_model=16,
        n_layers=2,
        n_heads=4,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = BioMGPTForSequenceClassification(
        backbone=backbone,
        num_classes=2,
        dropout=0.0,
    )
    model.eval()

    species_ids = torch.tensor([
        [0, 1, 2],
        [0, 1, 2],
    ])
    abundance_bins = torch.tensor([
        [0, 0, 3],
        [4, 5, 6],
    ])
    attention_mask = torch.ones_like(species_ids)

    out = model(
        species_ids=species_ids,
        abundance_bins=abundance_bins,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    assert out["cls_attention"] is not None
    assert out["cls_attention"].shape == species_ids.shape

    # recreate the CLS attention averaging across all heads and layers and compare
    stacked_attentions = torch.stack(out["attentions"], dim=0)
    cls_layers = stacked_attentions[:, :, :, 0, :]
    expected_cls_attention = cls_layers.mean(dim=2).mean(dim=0)[:, 1:]
    
    # breakpoint()

    assert torch.allclose(out["cls_attention"], expected_cls_attention)
    
if __name__ == "__main__":
    test_cls_attention_matches_manual_average()
    print("Test completed ✅ ")
