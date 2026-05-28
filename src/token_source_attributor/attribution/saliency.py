import torch
from captum.attr import Saliency

def saliency_ibd_batch(model, batch, device, target_class=1):
    """
    Quick and dirty Captum saliency for IBD logit.

    Returns:
        species_attr:   [B, S]
        abundance_attr: [B, S]
        total_attr:     [B, S]
        logits:         [B, C]
        probs:          [B, C]
        preds:          [B]
        confidence:     [B]
        entropy:        [B]
        surprise:       [B] if labels exist
        perplexity:     [B] if labels exist
    """
    # flip off the safety
    model.eval()
    
    # load data magazines
    species_ids = batch["species_ids"].to(device)
    abundance_bins = batch["abundance_bins"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # in case we want to do inference on unsupervised data or unlabeled samples
    labels = None
    if "label" in batch:
        labels = batch["label"].to(device)
        
    # 1. Add CLS exactly like normal model forward
    species_ids, abundance_bins_with_cls, attention_mask_with_cls = (
        model.backbone.prepend_cls(
            species_ids,
            abundance_bins,
            attention_mask,
        )
    )
    
    # 2. Build differentiable source embeddings
    species_emb, abundance_emb = model.backbone.build_source_embeddings(
        species_ids,
        abundance_bins_with_cls,
    )
    
    species_emb = species_emb.detach().requires_grad_(True)
    abundance_emb = abundance_emb.detach().requires_grad_(True)
    
    # 3. Captum-compatible forward
    def forward_fn(species_emb_input, abundance_emb_input):
        out = model.backbone.forward_from_components(
            species_emb=species_emb_input,
            abundance_emb=abundance_emb_input,
            attention_mask=attention_mask_with_cls,
            output_attentions=False,
        )

        logits = model.classifier(out["cls_state"])  # [B, C]

        # Captum Saliency wants a scalar per sample.
        # This is the IBD logit.
        return logits[:, target_class]
    
    # 4. Saliency attribution wrapper of the forward_fn
    saliency = Saliency(forward_fn)

    # captum forward pass
    species_attr_full, abundance_attr_full = saliency.attribute(
        inputs=(species_emb, abundance_emb),
    )

    # 5. Drop CLS token
    species_attr_full = species_attr_full[:, 1:, :]      # [B, S, D]
    abundance_attr_full = abundance_attr_full[:, 1:, :]  # [B, S, D]
    
    # 6. Collapse embedding dimension into token-level scores
    # abs() because saliency is gradient magnitude-ish; first pass: "how sensitive"
    
    # For directional attribution later, can use signed gradients:
    species_attr = species_attr_full.abs().sum(dim=-1)       # [B, S]
    abundance_attr = abundance_attr_full.abs().sum(dim=-1)   # [B, S]
    total_attr = species_attr + abundance_attr               # [B, S]
    
    # 7. Also return inference metrics
    with torch.no_grad():
        out = model(
            species_ids=batch["species_ids"].to(device),
            abundance_bins=batch["abundance_bins"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels,
        )
        
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confidence = probs.max(dim=1).values
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

        surprise = None
        perplexity = None
        
        if labels is not None:
            surprise = -torch.log(
                probs[torch.arange(labels.size(0), device=device), labels] + 1e-12
            )
            perplexity = torch.exp(surprise)
            
        # results are per batch inference
        return {
            "species_attr": species_attr,       # [B, S]
            "abundance_attr": abundance_attr,   # [B, S]
            "total_attr": total_attr,           # [B, S]
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "confidence": confidence,
            "entropy": entropy,
            "surprise": surprise,
            "perplexity": perplexity,
        }   