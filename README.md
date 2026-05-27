# token-source-attribution
A lightweight framework for attributing transformer predictions back to the individual sources that compose each input token.  It helps separate attribution across these components so model explanations can show not only which token mattered, but which part of the token representation drove the prediction.

For example, it can isolate the loss gradients wrt species identity embeddings and abundance embeddings in microbiome models.  More broadly, in multimodal or multi-source models, it can separate attribution across components such as text, image, metadata, or abundance-derived features.

## Installation



## Development

## Installation
To install the package, run the following command in your terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Model - Transformer Encoder Model training and evaluation
Transformer encoder MLM pretrained model and fine tuned classification head is based on the biomeGPT paper.
https://www.biorxiv.org/content/10.64898/2026.01.05.697599v1.full.pdf

#### Data
Shotgun metagenomic sequencing data profiled by MetaPhlAn (Metagenomic Phylogenetic Analysis) fetched from the curatedMetagenomicData R package. 27K stool samples used to train the transformer encoder model.

Installation instructions for fetching data

1) https://github.com/waldronlab/curatedMetagenomicDataTerminal

2) Run the following script to fetch the data and save it in the correct format for training the transformer encoder model.

```bash
chmod +x data/fetch_stool_species.R

./data/fetch_stool_species.R
```