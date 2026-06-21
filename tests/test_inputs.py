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

BATCH_SIZE = 32

device = "cpu"
dataset_path = "src/token_source_attributor/data/classifier_stool_binned.tsv"
checkpoint_path = "checkpoints_classifier/biomgpt_classify_1st_run_val-loss-0.21_val-acc-0.93acc.pt"
# target_class = 1 using log-odds now

dataset = BinnedMicrobiomeClassificationDataset(
    path=dataset_path,
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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

output_path = Path("tests/test_IG_output_unit.jsonl")
output_path.write_text("", encoding="utf-8")
with output_path.open("a", encoding="utf-8") as output_file:
    output_file.write(json.dumps({
        "record_type": "run_metadata",
        "dataset_path": dataset_path,
        "checkpoint_path": checkpoint_path,
        # "target_class": target_class, using log-odds now
    }) + "\n")

embedding_output_dir = Path("tests/test_IG_unit_embeddings")
embedding_output_dir.mkdir(exist_ok=True)

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
    if batch_index >= len(batch_records):
        assert False, f"DataLoader produced batch {batch_index}, but JSONL only has {len(batch_records)} records. 1:1 mapping broken."
    # Get the corresponding metadata for this batch
    record = batch_records[batch_index]
    if not record:
        assert False, "batches from dataloader should be exact same as batch_records"
    
    result = ig_inputs_batch(
        model=model,
        batch=batch,
        device=device,
    )
    
    abundance_embed_no_cls = result['abundance_embed_no_cls']
    abundance_baseline_no_cls = result['abundance_baseline_no_cls']
    
    dx = abundance_embed_no_cls - abundance_baseline_no_cls # [B,S,H] [B,1662,512]
    
    B,S,H = dx.shape
    assert S == 1662 # no [CLS]
    assert H == 512
    
    page_offset = batch_index * BATCH_SIZE
    
    # sample_nums
    kept_samples_pos = torch.tensor(record['true_positive_embedding_sample_numbers'], dtype=torch.int)
    kept_samples_neg = torch.tensor(record['true_negative_embedding_sample_numbers'], dtype=torch.int)
    
    # tp indices for batch
    tp_batch_indices = kept_samples_pos - page_offset

    # tn indices for batch
    tn_batch_indices = kept_samples_neg - page_offset
    
    # load corresponding IG embeddings for batch
    if tp_batch_indices.numel() != 0:
        file_path = f"tests/test_IG_embeddings/batch_{batch_index:03d}_true_positive_total_attr_embed.pt"
        tp_tensor = torch.load(file_path, map_location="cpu")
        print(tp_tensor.shape)
        
        # remove dx to get unit signal 
        tp_dx = dx[tp_batch_indices]
        
        # IG = ∫f'(x)dx, IG/dx = ∫f'(x),  dx = 0, IG = 0, 0 / 1e-9 = 0 -> ∫f'(x) = 0
        tp_unit_signal_emb = tp_tensor / (tp_dx + 1e-9)
        
        tp_unit_signal_tok = tp_unit_signal_emb.sum(axis=-1) #[B,S,H] -> [B,S]
        
        # save to jsonl for batch_index, sample num and token
        # save as ig_species_plus_abundance_unit
        sample_nums_pos = tp_batch_indices + page_offset
        
        # loop through sample nums in batch
        for i in range(sample_nums_pos.size(0)):
            # refernce the sample tokens
            global_num = sample_nums_pos[i].item()
            # Find the specific sample dict that matches this global number
            sample = next((s for s in record['samples'] if s["sample_number"] == global_num), None)
            
            if sample is None:
                print(f"Warning: Could not find global sample {global_num} in JSON map.")
                continue
            
            tok_tensor = tp_unit_signal_tok[i,:] # [S] 1662
            # for each sample token write unit abundance to it
            for j in range(len(sample['tokens'])):
                sample['tokens'][j]['ig_species_plus_abundance_unit'] = tok_tensor[j].item()
            
        
        # save tp_unit_signal to batch until signal pt file
        tp_embed_output_path = embedding_output_dir / f"batch_{batch_index:03d}_true_positive_total_attr_embed_unit.pt"
        torch.save(tp_unit_signal_emb, tp_embed_output_path)
        
    # load corresponding IG embeddings for batch
    if tn_batch_indices.numel() != 0:
        file_path = f"tests/test_IG_embeddings/batch_{batch_index:03d}_true_negative_total_attr_embed.pt"
        tn_tensor = torch.load(file_path, map_location="cpu")
        print(tn_tensor.shape)
        
        # remove dx to get unit signal 
        tn_dx = dx[tn_batch_indices]
        
        tn_unit_signal_emb = tn_tensor / (tn_dx + 1e-9)
        
        tn_unit_signal_tok = tn_unit_signal_emb.sum(axis=-1)
        
        # save to jsonl for batch and sample num
        sample_nums_neg = tn_batch_indices + page_offset
        
        # loop through here and save to the corresponding json
        # loop through sample nums in batch
        for i in range(sample_nums_neg.size(0)):
            # refernce the sample tokens
            global_num = sample_nums_neg[i].item()
            # Find the specific sample dict that matches this global number
            sample = next((s for s in record['samples'] if s["sample_number"] == global_num), None)
            
            if sample is None:
                print(f"Warning: Could not find global sample {global_num} in JSON map.")
                continue
            
            tok_tensor = tn_unit_signal_tok[i,:] # [S] 1662
            # for each sample token write unit abundance to it
            for j in range(len(sample['tokens'])):
                sample['tokens'][j]['ig_species_plus_abundance_unit'] = tok_tensor[j].item()
        
        # save tn_unit_signal to batch until signal pt file
      
        tn_embed_output_path = embedding_output_dir / f"batch_{batch_index:03d}_true_negative_total_attr_embed_unit.pt"
        torch.save(tn_unit_signal_emb, tn_embed_output_path)
        
    # write this batch to a new jsonl file
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record) + "\n")
        
    
    
    
     
