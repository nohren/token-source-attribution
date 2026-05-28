import torch
import torch.nn as nn
import torch.nn.functional as F
from token_source_attributor.data.data_preprocessing import bin_microbiome_sample


class BioMGPTEncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, output_attentions=False):
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=output_attentions,
            average_attn_weights=False,
        )

        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x, attn_weights if output_attentions else None


class BioMGPTEncoderBackbone(nn.Module):
    """
    BioMGPT-style BERT encoder backbone.

    Token = LN(species_embedding) + LN(abundance_mlp(abundance_bin))

    <cls> is represented as:
        species_id = num_species
        abundance_bin = 0

    No positional embeddings.
    """

    def __init__(
        self,
        num_species: int,
        max_bin: int = 50,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_species = num_species
        self.cls_species_id = num_species
        self.num_species_with_cls = num_species + 1

        self.max_bin = max_bin
        self.mask_bin_id = max_bin + 1
        self.d_model = d_model

        self.species_embedding = nn.Embedding(
            self.num_species_with_cls,
            d_model,
        )

        self.abundance_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.species_ln = nn.LayerNorm(d_model)
        self.abundance_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            BioMGPTEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

        nn.init.normal_(self.species_embedding.weight, std=0.02)

    def prepend_cls(self, species_ids, abundance_bins, attention_mask=None):
        '''
        Prepend <cls> token to the input sequences.
        B is batch size, S is sequence length (number of species per sample)
        species_ids: [B, S]
        abundance_bins: [B, S]

        species_ids =
            [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            ]

        abundance_bins =
            [
            [0, 12, 30, 0, 5],
            [4, 0, 50, 1, 0],
            [0, 0, 9, 20, 3],
            ]
        '''
        B = species_ids.size(0)

        cls_species = torch.full(
            (B, 1),
            fill_value=self.cls_species_id,
            dtype=species_ids.dtype,
            device=species_ids.device,
        )

        cls_abundance = torch.zeros(
            (B, 1),
            dtype=abundance_bins.dtype,
            device=abundance_bins.device,
        )

        species_ids = torch.cat([cls_species, species_ids], dim=1)
        abundance_bins = torch.cat([cls_abundance, abundance_bins], dim=1) 

        if attention_mask is not None:
            cls_mask = torch.ones(
                (B, 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        return species_ids, abundance_bins, attention_mask

    def build_source_embeddings(self, species_ids, abundance_bins):
        species_emb = self.species_embedding(species_ids)

        abundance_float = abundance_bins.float().unsqueeze(-1)

        # Optional but recommended: keep numeric input scale small.
        # paper does not specify this. Possibly optimization if backed by research.
        # abundance_float = abundance_float / float(self.max_bin)
        
        # convert abundance_bin scalar -> embedding
        abundance_emb = self.abundance_mlp(abundance_float)

        return species_emb, abundance_emb

    def compose_token_embedding(self, species_emb, abundance_emb):
        return self.species_ln(species_emb) + self.abundance_ln(abundance_emb)

    def forward_from_components(
        self,
        species_emb,
        abundance_emb,
        attention_mask=None,
        output_attentions=False,
    ):
        x = self.compose_token_embedding(species_emb, abundance_emb)
        x = self.dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        attentions = []

        for layer in self.layers:
            x, attn = layer(
                x,
                key_padding_mask=key_padding_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attentions.append(attn)

        x = self.final_ln(x)

        return {
            "last_hidden_state": x,       # [B, S+1, D]
            "cls_state": x[:, 0, :],      # [B, D] first token in the sequence
            "token_states": x[:, 1:, :],  # [B, S, D]
            "attentions": attentions,
        }

    def forward(
        self,
        species_ids,
        abundance_bins,
        attention_mask=None,
        output_attentions=False,
    ):
        # first prepend cls token and adjust attention mask
        species_ids, abundance_bins, attention_mask = self.prepend_cls(
            species_ids,
            abundance_bins,
            attention_mask,
        )

        # second build embeddings
        species_emb, abundance_emb = self.build_source_embeddings(
            species_ids,
            abundance_bins,
        )

        # finally compose token embeddings and pass through layers
        return self.forward_from_components(
            species_emb,
            abundance_emb,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

class BioMGPTForMaskedAbundanceModeling(nn.Module):
    """
    Foundation model pretraining head.

    Predicts original abundance bin values at masked nonzero positions.
    Uses MSE, matching the paper description.
    """

    def __init__(self, backbone: BioMGPTEncoderBackbone):
        super().__init__()
        self.backbone = backbone

        self.abundance_head = nn.Sequential(
            nn.Linear(backbone.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        species_ids,
        abundance_bins_masked,
        attention_mask=None,
        mlm_labels=None,
        output_attentions=False,
    ):
        out = self.backbone(
            species_ids=species_ids,
            abundance_bins=abundance_bins_masked,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # hidden states h_0,h_1, ... , h_t
        token_states = out["token_states"]  # excludes <cls>, [B, S, D]
        pred_bins = self.abundance_head(token_states).squeeze(-1)  # [B, S]

        loss = None
        if mlm_labels is not None:
            mask = mlm_labels != -100

            if mask.any():
                loss = F.mse_loss(
                    pred_bins[mask],
                    mlm_labels[mask].float(),
                )
            else:
                print('MLM labels are none!')
                loss = torch.tensor(0.0, device=pred_bins.device)

        return {
            "loss": loss,
            "pred_bins": pred_bins,
            "last_hidden_state": out["last_hidden_state"],
            "cls_state": out["cls_state"],
            "token_states": out["token_states"],
            "attentions": out["attentions"],
        }


class BioMGPTForSequenceClassification(nn.Module):
    """
    Later fine-tuning model.

    Use this only after the backbone/foundation model is pretrained.
    """

    def __init__(
        self,
        backbone: BioMGPTEncoderBackbone,
        num_classes: int,
        classifier_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone

        hidden_dim = classifier_hidden_dim or backbone.d_model

        # Classification head on top of transformer encoder backbone
        # receives CLS token representation h_[CLS]
        self.classifier = nn.Sequential(
            nn.Linear(backbone.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        species_ids,
        abundance_bins,
        attention_mask=None,
        labels=None,
        output_attentions=False,
    ):
        out = self.backbone(
            species_ids=species_ids,
            abundance_bins=abundance_bins,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # take [CLS] out of backbone and place into classifier as (B, D)
        # get out logits of shape (B, num_classes)
        logits = self.classifier(out["cls_state"])

        loss = None
        if labels is not None:
            # 2 classes
            loss = F.cross_entropy(logits, labels)
        else:
            print("labels are not defined!")

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": out["last_hidden_state"],
            "cls_state": out["cls_state"],
            "token_states": out["token_states"],
            "attentions": out["attentions"],
        }
    

#NOTE, maybe TODO. this is stochastic, for samples with large amount of non zero values we may not get adequate training signal, since this relies on law of large numbers.  Unsure if its a biased or unbiased stochastic algorithm.
# in the future can change to exact-k sampling if this is biased. if unbiased then keep it.
def mask_nonzero_abundance_bins(abundance_bins, mask_bin_id, mask_prob=0.25):
    """
    abundance_bins: [B, S], values 0..50
    mask_bin_id: special input value, e.g. 51

    Returns:
        abundance_bins_masked: [B, S]
        mlm_labels: [B, S], original bins at masked positions, -100 elsewhere
        masked_positions: [B, S], boolean
    """

    nonzero = abundance_bins > 0
    random_select = torch.rand_like(abundance_bins.float()) < mask_prob

    masked_positions = nonzero & random_select

    mlm_labels = torch.full_like(abundance_bins, -100)
    mlm_labels[masked_positions] = abundance_bins[masked_positions]

    abundance_bins_masked = abundance_bins.clone()
    abundance_bins_masked[masked_positions] = mask_bin_id

    return abundance_bins_masked, mlm_labels, masked_positions