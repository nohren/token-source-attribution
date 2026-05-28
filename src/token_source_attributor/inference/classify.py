import torch
import pandas as pd

def classifier_inference(model, dataloader, device):
    model.eval()
    rows = []
    
    with torch.no_grad():
        for batch in dataloader:
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
            
            logits = out["logits"]              # [B, C]
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # how confident is the model in its prediction?
            confidence = probs.max(dim=1).values
            # multinomial entropy
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            
            surprise = -torch.log(
                probs[torch.arange(labels.size(0), device=device), labels] + 1e-12
            )
            
            perplexity = torch.exp(surprise)
            
            for i in range(labels.size(0)):
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "study_id": batch["study_id"][i],
                    "disease": batch["disease"][i],
                    "label": labels[i].item(),
                    "pred": preds[i].item(),
                    "prob_healthy": probs[i, 0].item(),
                    "prob_ibd": probs[i, 1].item(),
                    "confidence": confidence[i].item(),
                    "entropy": entropy[i].item(),
                    "surprise": surprise[i].item(),
                    "perplexity": perplexity[i].item(),
                    "correct": preds[i].item() == labels[i].item(),
                })

    return pd.DataFrame(rows)