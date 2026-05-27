import torch
from token_source_attributor.data.dataset import BinnedMicrobiomeDataset, BinnedMicrobiomeClassificationDataset

# MetaPhlAn clades sequences must be in the exact same order and be the exact same clades between pretrained data, and classification data since the embedding space is defined by the pretrained vocab and the species ids are given by that ordering

# vocab set tupple (species name, species id) for pretrained data, and we will check that the classification data has the same set of species names and same ordering (same species ids)


def test_data_loading():

    dataset = BinnedMicrobiomeDataset(
        path="src/token_source_attributor/data/bert_pretraining_stool_binned.tsv",
    )
    num_species = dataset.num_species
    species_ids = set(zip(dataset.species_ids.tolist(), dataset.species_cols))
    sample = dataset[19380]

    print("Sample:", sample)
    print("Binned vector:", sample["abundance_bins"])
    assert len(sample["abundance_bins"]) == len(sample["species_ids"])
    assert dataset.num_species == len(sample["abundance_bins"])
    print("num species:", dataset.num_species)
    print("Test passed! ✅")
    return num_species, species_ids

def test_data_loading_classification(num_species, species_ids):
    dataset = BinnedMicrobiomeClassificationDataset(
        path="src/token_source_attributor/data/classifier_stool_binned.tsv",
    )

    # check the data for continuity of embedding space
    assert num_species == dataset.num_species, f"Expected num species {num_species} same as pretrained vocab, got {dataset.num_species}"
    assert species_ids == set(zip(dataset.species_ids.tolist(), dataset.species_cols)), f"Expected species ids {species_ids} same as pretrained vocab, got {set(zip(dataset.species_ids.tolist(), dataset.species_cols))}"
    
    sample = dataset[4]

    print("Sample:", sample)
    print("Binned vector:", sample["abundance_bins"])
    assert len(sample["abundance_bins"]) == len(sample["species_ids"])
    assert dataset.num_species == len(sample["abundance_bins"])
    print("num species:", dataset.num_species)
    print("Test passed! ✅")

if __name__ == "__main__":
    num_species, species_ids = test_data_loading()
    test_data_loading_classification(num_species, species_ids)