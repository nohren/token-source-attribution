#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(dplyr)
  library(purrr)
  library(tibble)
})

set.seed(42)

# -----------------------------
# Config

#Pretraining species_vocab.txt is the source of truth.
#Fine-tuning data must align to that exact column order.
#Missing species get 0.
#New unseen species are dropped.
# -----------------------------

output_file <- "ibd_vs_healthy_stool_matrix.tsv"
species_vocab_file <- "ibd_vs_healthy_species_vocab.txt"
healthy_sample_target <- 4000

# 1 = IBD, 0 = healthy
positive_label <- "IBD"
negative_label <- "healthy"

# -----------------------------
# Gather all relative abundance datasets
# -----------------------------

all_resources <- curatedMetagenomicData(".relative_abundance", dryrun = TRUE)
target_studies <- all_resources[grepl("\\.relative_abundance", all_resources)]

cat("Starting extraction of IBD vs healthy stool samples...\n")
cat("Number of target studies:", length(target_studies), "\n\n")

# -----------------------------
# PASS 0: Collect eligible metadata
# -----------------------------

all_metadata_rows <- list()

for (study in target_studies) {
  cat("Scanning metadata:", study, "\n")

  se_list <- tryCatch({
    curatedMetagenomicData(study, dryrun = FALSE, counts = FALSE)
  }, error = function(e) {
    cat("  Skipping due to error:", conditionMessage(e), "\n")
    NULL
  })

  if (is.null(se_list) || length(se_list) == 0) next

  se <- se_list[[1]]
  metadata <- as.data.frame(colData(se))

  if (!"body_site" %in% colnames(metadata)) next
  if (!"disease" %in% colnames(metadata)) next

  selected_metadata <- metadata %>%
    rownames_to_column("sample_id") %>%
    filter(body_site == "stool") %>%
    filter(!is.na(disease)) %>%
    filter(disease %in% c(positive_label, negative_label)) %>%
    mutate(
      study_id = study,
      Label = ifelse(disease == positive_label, 1L, 0L)
    ) %>%
    select(study_id, sample_id, disease, Label)

  if (nrow(selected_metadata) > 0) {
    all_metadata_rows[[length(all_metadata_rows) + 1]] <- selected_metadata
    cat("  Eligible samples:", nrow(selected_metadata), "\n")
  }

  rm(se, se_list, metadata, selected_metadata)
  gc(verbose = FALSE)
}

if (length(all_metadata_rows) == 0) {
  stop("No eligible IBD/healthy stool samples found.")
}

eligible_metadata <- bind_rows(all_metadata_rows)

ibd_metadata <- eligible_metadata %>%
  filter(disease == positive_label)

healthy_metadata <- eligible_metadata %>%
  filter(disease == negative_label)

cat("\nEligible sample counts before healthy downsampling:\n")
cat("  IBD:", nrow(ibd_metadata), "\n")
cat("  healthy:", nrow(healthy_metadata), "\n")

if (nrow(healthy_metadata) > healthy_sample_target) {
  healthy_metadata <- healthy_metadata %>%
    slice_sample(n = healthy_sample_target)
}

selected_metadata_all <- bind_rows(ibd_metadata, healthy_metadata) %>%
  arrange(study_id, sample_id)

cat("\nSelected final sample counts:\n")
cat("  IBD:", sum(selected_metadata_all$disease == positive_label), "\n")
cat("  healthy:", sum(selected_metadata_all$disease == negative_label), "\n")
cat("  total:", nrow(selected_metadata_all), "\n\n")

cat("Selected samples by disease:\n")
selected_disease_counts <- selected_metadata_all %>%
  count(disease, Label, sort = TRUE) %>%
  as_tibble()

print(selected_disease_counts, n = Inf)

cat("\nSelected samples by study and disease:\n")
selected_study_disease_counts <- selected_metadata_all %>%
  count(study_id, disease, sort = TRUE) %>%
  as_tibble()

print(selected_study_disease_counts, n = 100)

# Split selected metadata by study for easy lookup later
selected_by_study <- split(selected_metadata_all, selected_metadata_all$study_id)

# -----------------------------
# PASS 1: Load pretrained species vocabulary
# -----------------------------

species_vocab_file <- "species_vocab.txt"

if (!file.exists(species_vocab_file)) {
  stop(paste("Missing pretrained species vocab file:", species_vocab_file))
}

all_species <- readLines(species_vocab_file)

cat("\nLoaded pretrained species vocabulary size:", length(all_species), "\n")
cat("Using exact pretrained species order from:", species_vocab_file, "\n\n")

# -----------------------------
# PASS 2: Align each selected study to global vocabulary and write matrix
# -----------------------------

cat("PASS 2: Writing aligned IBD vs healthy stool matrix...\n")

is_first_write <- TRUE

for (study in names(selected_by_study)) {
  cat("Processing study:", study, "\n")

  study_selected_metadata <- selected_by_study[[study]]
  selected_sample_ids <- study_selected_metadata$sample_id

  se_list <- tryCatch({
    curatedMetagenomicData(study, dryrun = FALSE, counts = FALSE)
  }, error = function(e) {
    cat("  Skipping due to error:", conditionMessage(e), "\n")
    NULL
  })

  if (is.null(se_list) || length(se_list) == 0) next

  se <- se_list[[1]]

  se_selected <- se[, colnames(se) %in% selected_sample_ids, drop = FALSE]

  if (ncol(se_selected) == 0) {
    cat("  No selected columns found in this study, skipping\n")
    next
  }

  abundance_matrix <- assay(se_selected)

  species_rows <- grep("s__", rownames(abundance_matrix), value = TRUE)
  species_rows <- species_rows[!grepl("t__", species_rows)]

  if (length(species_rows) == 0) {
    cat("  No species rows found, skipping\n")
    next
  }

  matrix_species <- abundance_matrix[species_rows, , drop = FALSE]

  # Keep full MetaPhlAn clade strings as stable feature IDs
  rownames(matrix_species) <- species_rows

  # Transpose: rows = samples, columns = species
  pivoted_matrix <- t(matrix_species)
  df_pivoted <- as.data.frame(pivoted_matrix)

  # Move Sample IDs into explicit column
  df_pivoted <- rownames_to_column(df_pivoted, "Sample_ID")

  # Add metadata columns
  df_pivoted$Study_ID <- study

  df_pivoted <- df_pivoted %>%
    left_join(
      study_selected_metadata %>%
        select(sample_id, disease, Label),
      by = c("Sample_ID" = "sample_id")
    ) %>%
    rename(Disease = disease)

  # Add missing species columns as zero
  missing_species <- setdiff(all_species, colnames(df_pivoted))

  if (length(missing_species) > 0) {
    df_pivoted[missing_species] <- 0
  }

  # Reorder columns to fixed global order
  df_pivoted <- df_pivoted[, c("Study_ID", "Sample_ID", "Disease", "Label", all_species), drop = FALSE]

  # Optional safety check
  if (any(is.na(df_pivoted$Disease)) || any(is.na(df_pivoted$Label))) {
    stop(paste("Missing Disease/Label after join in study:", study))
  }

  cat("  Writing samples:", nrow(df_pivoted), "\n")

  if (is_first_write) {
    write.table(
      df_pivoted,
      file = output_file,
      sep = "\t",
      row.names = FALSE,
      col.names = TRUE,
      quote = FALSE
    )
    is_first_write <- FALSE
  } else {
    write.table(
      df_pivoted,
      file = output_file,
      sep = "\t",
      append = TRUE,
      row.names = FALSE,
      col.names = FALSE,
      quote = FALSE
    )
  }

  rm(se, se_list, se_selected, abundance_matrix, matrix_species, df_pivoted)
  gc(verbose = FALSE)
}

cat("\nIBD vs healthy stool matrix generation complete!\n")
cat("Wrote matrix:", output_file, "\n")
cat("Wrote species vocab:", species_vocab_file, "\n")