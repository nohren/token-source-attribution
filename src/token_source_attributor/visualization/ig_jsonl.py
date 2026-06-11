from __future__ import annotations

import html
import json
import random
from pathlib import Path
from typing import Any

def display_random_tp_tn_token_charts(
    jsonl_path: str | Path = "tests/test_IG_output.jsonl",
    species_vocab_path: str | Path = "src/token_source_attributor/data/species_vocab.txt",
    samples_per_class: int = 3,
    top_k: int = 10,
    seed: int | None = None,
    display_output: bool = True,
) -> dict[str, Any]:
    species_vocab = _load_species_vocab(species_vocab_path)
    sampled_records = _sample_records_by_prediction_type(
        jsonl_path=jsonl_path,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    dashboard_html = _build_dashboard_html(
        sampled_records=sampled_records,
        species_vocab=species_vocab,
        top_k=top_k,
    )

    if display_output:
        from IPython.display import HTML, display

        display(HTML(dashboard_html))

    return {
        "true_positive": sampled_records["true_positive"],
        "true_negative": sampled_records["true_negative"],
        "html": dashboard_html,
    }


def _load_species_vocab(species_vocab_path: str | Path) -> list[str]:
    with Path(species_vocab_path).open(encoding="utf-8") as vocab_file:
        return [line.strip() for line in vocab_file if line.strip()]


def _sample_records_by_prediction_type(
    jsonl_path: str | Path,
    samples_per_class: int,
    seed: int | None,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    reservoirs = {
        "true_positive": [],
        "true_negative": [],
    }
    seen_counts = {
        "true_positive": 0,
        "true_negative": 0,
    }

    with Path(jsonl_path).open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if record.get("record_type") != "batch":
                continue

            for sample in record.get("samples", []):
                prediction_type = sample.get("prediction_type")
                if prediction_type not in reservoirs:
                    continue

                seen_counts[prediction_type] += 1
                kept_sample = {
                    "batch_index": record.get("batch_index"),
                    **sample,
                }
                reservoir = reservoirs[prediction_type]

                if len(reservoir) < samples_per_class:
                    reservoir.append(kept_sample)
                    continue

                replacement_index = rng.randint(1, seen_counts[prediction_type])
                if replacement_index <= samples_per_class:
                    reservoir[replacement_index - 1] = kept_sample

    return reservoirs


def _build_dashboard_html(
    sampled_records: dict[str, list[dict[str, Any]]],
    species_vocab: list[str],
    top_k: int,
) -> str:
    sections = []
    for prediction_type, title in (
        ("true_positive", "True Positive"),
        ("true_negative", "True Negative"),
    ):
        cards = sampled_records[prediction_type]
        card_html = "".join(
            _build_sample_card_html(sample=sample, species_vocab=species_vocab, top_k=top_k)
            for sample in cards
        )
        if not card_html:
            card_html = '<p style="margin:0;color:#666;">No samples found.</p>'

        sections.append(
            f"""
            <section style="margin-bottom:32px;">
              <h2 style="margin:0 0 12px 0;font-family:sans-serif;">{title}</h2>
              <div style="display:grid;gap:16px;">{card_html}</div>
            </section>
            """
        )

    return f"""
    <div style="font-family:sans-serif;line-height:1.5;">
      <div style="margin-bottom:20px;">
        <h1 style="margin:0 0 6px 0;">Random TP/TN IG Token Visualizations</h1>
        <p style="margin:0;color:#555;">
          Each sample shows the top {top_k} tokens sorted by abundance. Color intensity is normalized within each row.
        </p>
      </div>
      {''.join(sections)}
    </div>
    """


def _build_sample_card_html(
    sample: dict[str, Any],
    species_vocab: list[str],
    top_k: int,
) -> str:
    top_tokens = sorted(
        sample["tokens"],
        key=lambda token: (
            -int(token["abundance_bin"]),
            -abs(float(token["ig_species_plus_abundance"])),
            int(token["token_index"]),
        ),
    )[:top_k]

    metadata = (
        f"sample={html.escape(str(sample['sample_id']))} | "
        f"study={html.escape(str(sample['study_id']))} | "
        f"disease={html.escape(str(sample['disease']))} | "
        f"label={sample['label']} pred={sample['pred']} | "
        f"prob_ibd={float(sample['prob_ibd']):.3f} | "
        f"batch={sample['batch_index']}"
    )

    return f"""
    <article style="border:1px solid #d8d8d8;border-radius:12px;padding:16px;background:#fcfcfc;">
      <div style="margin-bottom:10px;">
        <div style="font-weight:700;">{html.escape(sample['prediction_type']).replace('_', ' ').title()}</div>
        <div style="color:#555;font-size:13px;">{metadata}</div>
      </div>
      <div style="display:grid;gap:10px;">
        {_build_metric_row_html('Chart 1: Total IG', top_tokens, species_vocab, 'ig_species_plus_abundance', (196, 30, 58))}
        {_build_metric_row_html('Chart 1: Attention', top_tokens, species_vocab, 'cls_score', (37, 99, 235))}
        {_build_metric_row_html('Chart 2: Species IG', top_tokens, species_vocab, 'ig_species', (36, 138, 61))}
        {_build_metric_row_html('Chart 2: Abundance IG', top_tokens, species_vocab, 'ig_abundance', (126, 34, 206))}
      </div>
    </article>
    """


def _build_metric_row_html(
    title: str,
    tokens: list[dict[str, Any]],
    species_vocab: list[str],
    metric_key: str,
    rgb: tuple[int, int, int],
) -> str:
    max_abs_value = max(abs(float(token[metric_key])) for token in tokens) if tokens else 1.0
    if max_abs_value == 0:
        max_abs_value = 1.0

    token_spans = []
    for token in tokens:
        value = float(token[metric_key])
        intensity = abs(value) / max_abs_value
        alpha = 0.12 + (0.88 * intensity)
        label = _format_token_label(token, species_vocab)
        tooltip = (
            f"{title}: {value:.5f} | "
            f"species_ig={float(token['ig_species']):.5f} | "
            f"abundance_ig={float(token['ig_abundance']):.5f} | "
            f"attention={float(token['cls_score']):.5f}"
        )

        token_spans.append(
            f"""
            <span title="{html.escape(tooltip)}"
                  style="
                    display:inline-block;
                    margin:4px 6px 4px 0;
                    padding:4px 8px;
                    border-radius:999px;
                    background:rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.3f});
                    color:#111;
                    font-family:monospace;
                    font-size:12px;
                    white-space:nowrap;
                  ">
              {html.escape(label)}
            </span>
            """
        )

    return f"""
    <div>
      <div style="font-size:13px;font-weight:600;margin-bottom:4px;">{html.escape(title)}</div>
      <div>{''.join(token_spans)}</div>
    </div>
    """


def _format_token_label(token: dict[str, Any], species_vocab: list[str]) -> str:
    species_id = int(token["species_id"])
    species_name = species_vocab[species_id] if 0 <= species_id < len(species_vocab) else f"species_{species_id}"
    short_name = _get_species_name(species_name)
    if short_name.startswith("s__"):
        short_name = short_name[3:]
    short_name = short_name.replace("_", " ")
    return f"(s. {short_name}, {int(token['abundance_bin'])})"


def _get_species_name(clade_str: str) -> str:
    return clade_str.split("|")[-1]
