import torch
from captum.attr import IntegratedGradients

def ig_ibd_batch(model, batch, device, target_class=1):
    """
    Captum forward IG for IBD logit.
    
    TODO: Do we want to target the output diff IBD - healthy? How?

    Returns:
        species_attr:   [B, S]
        abundance_attr: [B, S]
        total_attr:     [B, S]
        [CLS] attention: [B,S]
        logits:         [B, C]
        probs:          [B, C]
        preds:          [B]
        confidence:     [B]
        entropy:        [B]
        surprise:       [B] if labels exist
        perplexity:     [B] if labels exist
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
    
    # cut the compute graph here in the backward
    # make these intermediate tensors the inputs
    species_emb = species_emb.detach().requires_grad_(True)
    abundance_emb = abundance_emb.detach().requires_grad_(True)
    
    # 3. Captum-compatible forward
    def forward_fn(species_emb_input, abundance_emb_input, attention_mask_input):
        out = model.backbone.forward_from_components(
            species_emb=species_emb_input,
            abundance_emb=abundance_emb_input,
            attention_mask=attention_mask_input,
            output_attentions=False
        )

        logits = model.classifier(out["cls_state"])  # [B, C]

        # Captum IG a scalar output unless target is specified
        # Using log odds, the difference of the
        # multinomial logits to prevent common mode variance
        # i.e feature logits arbitrarily scaled equally without altering the final probability distribution
        # just looking at target class you think something changed, but it didn't
        
        # ∂(IBD - Healthy)/∂x_i - if we wiggle this species tokens abundance input then ibd - healthy changes this much. where ibd - healthy is if the gradient is positive it drives toward ibd and if negative it drives toward healthy
        return logits[:, 1] - logits[:, 0]
    
    # 4. IG attribution wrapper of the forward_fn
    ig = IntegratedGradients(forward_fn)
    
    # breakpoint()
    
    # species embed is always the same since same species_ids always the same input to embedding
    # only thing that changes is abundance embedding
    # those are the actual words/tokens
    species_baseline = species_emb.clone()
    _, abundance_baseline = model.backbone.build_source_embeddings(
            species_ids_with_cls,
            torch.zeros_like(abundance_bins_with_cls),
        )

    # captum forward pass
    (species_attr_full, abundance_attr_full), delta = ig.attribute(
        inputs=(species_emb, abundance_emb),
        baselines=(
            species_baseline,
            abundance_baseline,
        ),
        additional_forward_args=(attention_mask_with_cls,),
        n_steps=150, # increase if delta sucks
        return_convergence_delta=True,
        # No need to provide target output, see the wrapped forward function chooses target logit already
        method='gausslegendre',
        internal_batch_size=128 # microbatches, memory expensive with B*n_steps, microbatch it to make forward B smaller
    )

    # 5. Drop sample CLS token from gradient attribution across batch
    # dont need it for data science portion
    species_attr_full_no_cls = species_attr_full[:, 1:, :]      # [B, S, H]
    abundance_attr_full_no_cls = abundance_attr_full[:, 1:, :]  # [B, S, H]
    # source IG = sum across feature level IG for source
    species_attr = species_attr_full_no_cls.sum(dim=-1)        # [B, S]
    abundance_attr = abundance_attr_full_no_cls.sum(dim=-1) # [B, S]
    # total
    total_attr = species_attr + abundance_attr # [B, S]

    # IG completeness diagnostics should be checked on the full inputs,
    # including the prepended CLS token, because Captum's delta is computed
    # against the full attribution tensors returned by attribute(...).
    # species_attr_full_sum = species_attr_full.flatten(1).sum(dim=1)
    # abundance_attr_full_sum = abundance_attr_full.flatten(1).sum(dim=1)
    # total_attr_full_sum = species_attr_full_sum + abundance_attr_full_sum

    # with torch.no_grad():
    #     target_logit = forward_fn(
    #         species_emb,
    #         abundance_emb,
    #         attention_mask_with_cls,
    #     )
    #     baseline_logit = forward_fn(
    #         species_baseline,
    #         abundance_baseline,
    #         attention_mask_with_cls,
    #     )
    #     logit_diff = target_logit - baseline_logit
    #     completeness_residual = total_attr_full_sum - logit_diff
    
    # Do one forward for inference metrics
    # Do it here since IG needs to run a number of times
    # expose attention [CLS] here and average across heads and layers
    with torch.no_grad():
        out = model(
            species_ids=batch["species_ids"].to(device),
            abundance_bins=batch["abundance_bins"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels,
            output_attentions=True,
        )
        
        logits = out["logits"]
        cls_attention = out["cls_attention"]
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
            "species_attr_signed": species_attr,       # [B, S]
            "abundance_attr_signed": abundance_attr,   # [B, S]
            "total_attr_signed": total_attr,  
            "species_attr_embed": species_attr_full_no_cls,
            "abundance_attr_embed": abundance_attr_full_no_cls,
            "total_attr_embed": species_attr_full_no_cls + abundance_attr_full_no_cls, # [B,S,H] k-means
            "batch_delta": delta,
            # "target_logit": target_logit,
            # "baseline_logit": baseline_logit,
            # "logit_diff": logit_diff,
            # "completeness_residual": completeness_residual,
            # [B, S]
            "logits": logits,                          # [B, S]
            "cls_attention": cls_attention,            # [B, S]
            "probs": probs,
            "preds": preds,
            "confidence": confidence,
            "entropy": entropy,
            "surprise": surprise,
            "perplexity": perplexity,
        }   
