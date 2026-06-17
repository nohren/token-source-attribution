import torch

@torch.no_grad()
def ig_inputs_batch(model, batch, device):
    """
    Returns:
        abundance_baseline: [B,S,H] x'
        abundance_embed: [B,S,H] x
        
        batch_level_diff = (abundance_embed - abundance_baseline)
    """
    model.eval()
    
    species_ids = batch["species_ids"].to(device)
    abundance_bins = batch["abundance_bins"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # in case we want to do inference on unsupervised data or unlabeled samples
    labels = None
    if "label" in batch:
        labels = batch["label"].to(device)
        
    # 1. Add CLS exactly like normal model forward
    species_ids_with_cls, abundance_bins_with_cls, attention_mask_with_cls = (
        model.backbone.prepend_cls(
            species_ids,
            abundance_bins,
            attention_mask,
        )
    )
    
    # 2. Build differentiable source embeddings
    # [B,S,H], [B,S,H]
    # remember every sample has all species
    # presence or absence is determined by abundance bins
    species_emb, abundance_emb = model.backbone.build_source_embeddings(
        species_ids_with_cls,
        abundance_bins_with_cls,
    )
    
    # species embed is always the same since same species_ids always the same input to embedding
    # only thing that changes is abundance embedding
    # those are the actual words/tokens
    species_baseline = species_emb.clone()
    _, abundance_baseline = model.backbone.build_source_embeddings(
            species_ids_with_cls,
            torch.zeros_like(abundance_bins_with_cls),
        )


    abundance_embed_no_cls = abundance_emb[:,1:,:].detach()
    abundance_baseline_no_cls = abundance_baseline[:,1:,:].detach()
     
    return {
        "abundance_embed_no_cls": abundance_embed_no_cls,       # [B, S, H]
        "abundance_baseline_no_cls": abundance_baseline_no_cls,   # [B, S, H]
    }   
