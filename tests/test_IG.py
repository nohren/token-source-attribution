import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from token_source_attributor.attribution.IG import ig_ibd_batch
from token_source_attributor.data.dataset import BinnedMicrobiomeClassificationDataset
from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForSequenceClassification,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "src/token_source_attributor/data/classifier_stool_binned.tsv"
checkpoint_path = "checkpoints_classifier/biomgpt_classify_1st_run_val-loss-0.21_val-acc-0.93acc.pt"
# target_class = 1 using log-odds now

dataset = BinnedMicrobiomeClassificationDataset(
    path=dataset_path,
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

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
    torch.load(
        checkpoint_path,
        map_location=device,
    )
)

embedding_output_dir = Path("tests/test_IG_embeddings")
embedding_output_dir.mkdir(exist_ok=True)


def build_token_records(batch, result, sample_in_batch_index):
    species_ids = batch["species_ids"][sample_in_batch_index]
    abundance_bins = batch["abundance_bins"][sample_in_batch_index]
    species_attr = result["species_attr_signed"][sample_in_batch_index]
    abundance_attr = result["abundance_attr_signed"][sample_in_batch_index]
    total_attr = result["total_attr_signed"][sample_in_batch_index]
    cls_attention = result["cls_attention"][sample_in_batch_index]

    token_records = []
    for token_index in range(species_ids.size(0)):
        token_records.append({
            "token_index": token_index,
            "species_id": int(species_ids[token_index].item()),
            "abundance_bin": int(abundance_bins[token_index].item()),
            "ig_species": float(species_attr[token_index].item()),
            "ig_abundance": float(abundance_attr[token_index].item()),
            "ig_species_plus_abundance": float(total_attr[token_index].item()),
            "cls_score": float(cls_attention[token_index].item()),
        })

    return token_records


def classify_prediction(label, pred):
    if label == 1 and pred == 1:
        return "true_positive"
    if label == 0 and pred == 0:
        return "true_negative"
    return None


output_path = Path("tests/test_IG_output.jsonl")
output_path.write_text("", encoding="utf-8")
with output_path.open("a", encoding="utf-8") as output_file:
    output_file.write(json.dumps({
        "record_type": "run_metadata",
        "dataset_path": dataset_path,
        "checkpoint_path": checkpoint_path,
        # "target_class": target_class, using log-odds now
    }) + "\n")

sample_number = 0

# batch by batch, do inference
for batch_index, batch in enumerate(loader):
    result = ig_ibd_batch(
        model=model,
        batch=batch,
        device=device,
        # not passing a target, using log-odds to provide a scalar logit_ibd - logit_healthy
    )
    
    print('Inference Results for Batch: ', batch_index, ' of ', len(loader))
    print('batch delta: ', result['batch_delta'])
    # print()
    # print('if low increase num_steps in IG.py')
    
    # print('Test')
    # print('logit diff: ', result['logit_diff'])
    # print('completeness_residual: ', result['completeness_residual'])

    batch_samples = []
    kept_sample_indices_tp = []
    kept_sample_numbers_tp = []
    kept_sample_indices_tn = []
    kept_sample_numbers_tn = []

    # sample by sample in batch results
    for sample_in_batch_index in range(batch["species_ids"].size(0)):
        label = int(batch["label"][sample_in_batch_index].item())
        pred = int(result["preds"][sample_in_batch_index].item())
        prediction_type = classify_prediction(label, pred)

        if prediction_type is None:
            sample_number += 1
            continue

        if prediction_type == "true_positive":
            kept_sample_indices_tp.append(sample_in_batch_index)
            kept_sample_numbers_tp.append(sample_number)
        elif prediction_type == "true_negative":
            kept_sample_indices_tn.append(sample_in_batch_index)
            kept_sample_numbers_tn.append(sample_number)

        batch_samples.append({
            "sample_number": sample_number,
            "sample_id": batch["sample_id"][sample_in_batch_index],
            "study_id": batch["study_id"][sample_in_batch_index],
            "disease": batch["disease"][sample_in_batch_index],
            "label": label,
            "pred": pred,
            "prediction_type": prediction_type,
            "prob_healthy": float(result["probs"][sample_in_batch_index, 0].item()),
            "prob_ibd": float(result["probs"][sample_in_batch_index, 1].item()),
            "confidence": float(result["confidence"][sample_in_batch_index].item()),
            "entropy": float(result["entropy"][sample_in_batch_index].item()),
            "tokens": build_token_records(batch, result, sample_in_batch_index),
        })

        sample_number += 1

    if batch_samples:
        tp_embedding_output_path = None
        tp_total_attr_embed_shape = None
        if kept_sample_indices_tp:
            batch_total_attr_embed_tp = result["total_attr_embed"][kept_sample_indices_tp].detach().cpu()
            tp_embedding_output_path = embedding_output_dir / f"batch_{batch_index:03d}_true_positive_total_attr_embed.pt"
            torch.save(batch_total_attr_embed_tp, tp_embedding_output_path)
            tp_total_attr_embed_shape = list(batch_total_attr_embed_tp.shape)

        tn_embedding_output_path = None
        tn_total_attr_embed_shape = None
        if kept_sample_indices_tn:
            batch_total_attr_embed_tn = result["total_attr_embed"][kept_sample_indices_tn].detach().cpu()
            tn_embedding_output_path = embedding_output_dir / f"batch_{batch_index:03d}_true_negative_total_attr_embed.pt"
            torch.save(batch_total_attr_embed_tn, tn_embedding_output_path)
            tn_total_attr_embed_shape = list(batch_total_attr_embed_tn.shape)

        batch_record = {
            "record_type": "batch",
            "batch_index": batch_index,
            "num_kept_samples": len(batch_samples),
            "num_true_positive_samples": len(kept_sample_indices_tp),
            "num_true_negative_samples": len(kept_sample_indices_tn),
            "true_positive_total_attr_embed_file": str(tp_embedding_output_path) if tp_embedding_output_path else None,
            "true_positive_total_attr_embed_shape": tp_total_attr_embed_shape,
            "true_positive_embedding_sample_numbers": kept_sample_numbers_tp,
            "true_negative_total_attr_embed_file": str(tn_embedding_output_path) if tn_embedding_output_path else None,
            "true_negative_total_attr_embed_shape": tn_total_attr_embed_shape,
            "true_negative_embedding_sample_numbers": kept_sample_numbers_tn,
            "samples": batch_samples,
        }
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(batch_record) + "\n")
    
    # # test 1 batch
    # break

print(f"Wrote IG JSONL output to {output_path}")
