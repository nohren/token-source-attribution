import pandas as pd
import numpy as np


def bin_microbiome_sample(abundance_vector, B=50):
    binned = np.zeros_like(abundance_vector, dtype=np.int16)

    nonzero_idx = np.where(abundance_vector > 0)[0]
    n = len(nonzero_idx)

    if n == 0:
        return binned

    sorted_idx = nonzero_idx[np.argsort(-abundance_vector[nonzero_idx])]

    if n >= B:
        chunks = np.array_split(sorted_idx, B)

        for chunk_i, chunk in enumerate(chunks):
            bin_value = B - chunk_i
            binned[chunk] = bin_value
    else:
        bin_values = np.linspace(B, 1, num=n).astype(np.int16)

        for idx, bin_value in zip(sorted_idx, bin_values):
            binned[idx] = bin_value

    return binned


def bin_abundance_file(
    input_file="bert_pretraining_stool_matrix.tsv",
    output_file="bert_pretraining_stool_binned.tsv",
    B=50,
    chunksize=512,
):
    first_write = True

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=chunksize):
        meta = chunk[["Study_ID", "Sample_ID"]]
        abundance = chunk.drop(columns=["Study_ID", "Sample_ID"])

        binned_rows = np.stack([
            bin_microbiome_sample(row.to_numpy(dtype=float), B=B)
            for _, row in abundance.iterrows()
        ])

        binned_df = pd.DataFrame(
            binned_rows,
            columns=abundance.columns,
        )

        out = pd.concat(
            [meta.reset_index(drop=True), binned_df.reset_index(drop=True)],
            axis=1,
        )

        out.to_csv(
            output_file,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

    print(f"Wrote binned file to {output_file}")


def main():
    bin_abundance_file()


if __name__ == "__main__":
    main()