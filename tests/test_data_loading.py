import torch
from token_source_attributor.data.dataset import BinnedMicrobiomeDataset

def test_data_loading():
    dataset = BinnedMicrobiomeDataset(
        path="src/token_source_attributor/data/bert_pretraining_stool_binned.tsv",
    )
    sample = dataset[19380]

    print("Sample:", sample)
    print("Abundance vector:", sample["abundance_bins"])
    print("Binned vector:", sample["abundance_bins"])
    assert len(sample["abundance_bins"]) == len(sample["species_ids"])
    assert dataset.num_species == len(sample["abundance_bins"])
    print("num species:", dataset.num_species)
    print("Test passed! ✅")

if __name__ == "__main__":
    test_data_loading()