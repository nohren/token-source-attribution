# token-source-attribution
A lightweight framework for attributing transformer predictions back to the individual sources that compose each input token.  It helps separate attribution across these components so model explanations can show not only which token mattered, but which part of the token representation drove the prediction.

For example, it can isolate the loss gradients wrt species identity embeddings and abundance embeddings in microbiome models.  More broadly, in multimodal or multi-source models, it can separate attribution across components such as text, image, metadata, or abundance-derived features.

## Installation



## Development

## Installation
To install the package, run the following command in your terminal:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Transformer Encoder Model training and evaluation

#### Data for model training and evaluation
Follow this for installation instructions 
https://github.com/waldronlab/curatedMetagenomicDataTerminal


```bash
chmod +x data/fetch_stool_species.R

./data/fetch_stool_species.R
```