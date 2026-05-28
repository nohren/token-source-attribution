import pandas as pd
import torch
from torch.utils.data import Dataset


class BinnedMicrobiomeDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path, sep="\t")

        self.meta_cols = ["Study_ID", "Sample_ID"]
        self.species_cols = [
            col for col in self.df.columns
            if col not in self.meta_cols
        ]

        self.num_species = len(self.species_cols)

        # Since every row has the same species vocabulary/order,
        # species_ids are just [0, 1, 2, ..., num_species - 1]
        self.species_ids = torch.arange(self.num_species, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    # NOTE get item for data loader is very important, as it defines how we read and process each sample
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        abundance_bins = row[self.species_cols].to_numpy(dtype="int64")
        abundance_bins = torch.tensor(abundance_bins, dtype=torch.long)

        attention_mask = torch.ones(self.num_species, dtype=torch.long)

        # this assumes a dense matrix design, lots of zeros and wasted compute, but it's simpler to implement and should be fine for now
        return {
            "species_ids": self.species_ids.clone(), # model will add one more for [CLS] token later
            "abundance_bins": abundance_bins,
            "attention_mask": attention_mask,
            "sample_id": row["Sample_ID"],
            "study_id": row["Study_ID"],
        }
    
class BinnedMicrobiomeClassificationDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path, sep="\t")

        self.meta_cols = ["Study_ID", "Sample_ID", "Disease", "Label"]
        # use later for attribution analysis to see which species are most important for predicting each disease
        self.species_cols = [
            col for col in self.df.columns
            if col not in self.meta_cols
        ]

        self.num_species = len(self.species_cols)

        # Since every row has the same species vocabulary/order,
        # species_ids are just [0, 1, 2, ..., num_species - 1]
        self.species_ids = torch.arange(self.num_species, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    # NOTE get item for data loader is very important, as it defines how we read and process each sample
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        abundance_bins = row[self.species_cols].to_numpy(dtype="int64")
        abundance_bins = torch.tensor(abundance_bins, dtype=torch.long)
        attention_mask = torch.ones(self.num_species, dtype=torch.long)
        label = torch.tensor(row["Label"], dtype=torch.long)


        
        return {
            "species_ids": self.species_ids.clone(), # model will add one more for [CLS] token later
            "abundance_bins": abundance_bins,
            "attention_mask": attention_mask,
            "sample_id": row["Sample_ID"],
            "study_id": row["Study_ID"],
            "disease": row["Disease"],
            "label": label
        }
        
def get_species_name(clade_str):
    return clade_str.split('|')[-1]