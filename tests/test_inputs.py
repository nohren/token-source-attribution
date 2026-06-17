import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from token_source_attributor.attribution.inputs import ig_inputs_batch
from token_source_attributor.data.dataset import BinnedMicrobiomeClassificationDataset
from token_source_attributor.models.biomgpt import (
    BioMGPTEncoderBackbone,
    BioMGPTForSequenceClassification,
)


device = "cpu"
dataset_path = "src/token_source_attributor/data/classifier_stool_binned.tsv"
checkpoint_path = "checkpoints_classifier/biomgpt_classify_1st_run_val-loss-0.21_val-acc-0.93acc.pt"
# target_class = 1 using log-odds now

dataset = BinnedMicrobiomeClassificationDataset(
    path=dataset_path,
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

for batch_index, batch in enumerate(loader):
    val = batch["species_ids"].size(0)
    breakpoint()



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

embedding_output_dir = Path("tests/test_input_embeddings")
embedding_output_dir.mkdir(exist_ok=True)

sample_number = 0

# 1. Load your master map
jsonl_file = "tests/test_IG_output.jsonl"
batch_records = []
with open(jsonl_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get("record_type") == "batch":
            batch_records.append(data)

# batch by batch, do inference
for batch_index, batch in enumerate(loader):
    
    # Get the corresponding metadata for this batch
    record = next((r for r in batch_records if r["batch_index"] == batch_index), None)
    if not record:
        continue # Skipped batch
    
    result = ig_inputs_batch(
        model=model,
        batch=batch,
        device=device,
    )
    
    abundance_embed_no_cls = result['abundance_embed_no_cls']
    abundance_baseline_no_cls = result['abundance_baseline_no_cls']
    
    dx = abundance_embed_no_cls - abundance_baseline_no_cls # [B,S,H] [B,1662,512]
    
    B,S,H = dx.shape
    assert S == 1662
    assert H == 512
    
    # sample_nums
    kept_samples_pos = torch.tensor(batch_records[batch_index]['true_positive_embedding_sample_numbers'], dtype=torch.int)
    kept_samples_neg = torch.tensor(batch_records[batch_index]['true_negative_embedding_sample_numbers'], dtype=torch.int)
    
    batch_start_sample_num = torch.cat([
        kept_samples_pos,
        kept_samples_neg
    ]).min()
    
    # tp indices for batch
    tp_batch_indices = kept_samples_pos - batch_start_sample_num

    # tn indices for batch
    tn_batch_indices = kept_samples_neg - batch_start_sample_num
    
    # load corresponding IG embeddings for batch
    if tp_batch_indices.numel != 0:
        file_path = f"test_IG_embeddings/batch_{batch_index:03d}_true_positive_total_attr_embed.pt"
        tp_tensor = torch.load(file_path, map_location="cpu")
        print(tp_tensor.shape)
        
        # remove dx to get unit signal 
        tp_dx = dx[tp_batch_indices]
        
        tp_unit_signal = tp_tensor / (tp_dx + 1e-9)
        
        # save to jsonl for batch and sample num
        sample_nums = tp_batch_indices + batch_start_sample_num
        
        # loop through here and save to the corresponding json
        
    # load corresponding IG embeddings for batch
    if tn_batch_indices.numel != 0:
        file_path = f"test_IG_embeddings/batch_{batch_index:03d}_true_negative_total_attr_embed.pt"
        tn_tensor = torch.load(file_path, map_location="cpu")
        print(tn_tensor.shape)
        
        # remove dx to get unit signal 
        tn_dx = dx[tn_batch_indices]
        
        tn_unit_signal = tn_tensor / (tn_dx + 1e-9)
        
        # save to jsonl for batch and sample num
        sample_nums = tn_batch_indices + batch_start_sample_num
        
        # loop through here and save to the corresponding json
        
        
    
    
    
    
    

    # match to jsonl samples, which ig embeddings are also matched to 
