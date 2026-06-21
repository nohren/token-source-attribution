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


embedding_output_dir = Path("tests/test_species_embeddings")
embedding_output_dir.mkdir(exist_ok=True)

# batch by batch, do inference
for batch_index, batch in enumerate(loader): 
    result = ig_inputs_batch(
        model=model,
        batch=batch,
        device=device,
    )
    
    species_emb_no_cls = result['species_emb_no_cls'] # [B,S,H]
   
    species_embed_output_path = embedding_output_dir / f"batch_{batch_index:03d}_species_embeddings.pt"
    torch.save(species_emb_no_cls, species_embed_output_path)
        
        
    
    
    
     
