#!/usr/bin/env Rscript

# Suppress messages and load libraries quietly
suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(dplyr)
  library(purrr)
})

# 1. Gather all relative abundance datasets
all_resources <- curatedMetagenomicData(".relative_abundance", dryrun = TRUE)
target_studies <- all_resources[grepl("\\.relative_abundance", all_resources)]

output_file <- "bert_pretraining_stool_matrix.tsv"
is_first_write <- TRUE

cat("Starting extraction of global stool samples for BERT pre-training...\n")

# -----------------------------
# PASS 1: Build global species vocabulary
# -----------------------------

all_species <- character()

for (study in target_studies) {
  se_list <- tryCatch({
    curatedMetagenomicData(study, dryrun = FALSE, counts = FALSE)
  }, error = function(e) NULL)

  if (is.null(se_list) || length(se_list) == 0) next
  se <- se_list[[1]]

  metadata <- as.data.frame(colData(se))
  if (!"body_site" %in% colnames(metadata)) next

  stool_ids <- metadata %>%
    tibble::rownames_to_column("sample_id") %>%
    filter(body_site == "stool") %>%
    pull(sample_id)

  if (length(stool_ids) == 0) next

  abundance_matrix <- assay(se[, colnames(se) %in% stool_ids, drop = FALSE])

  species_rows <- grep("s__", rownames(abundance_matrix), value = TRUE)
  species_rows <- species_rows[!grepl("t__", species_rows)]

  # IMPORTANT: keep full MetaPhlAn clade strings as stable feature IDs
  all_species <- union(all_species, species_rows)

  rm(se, se_list, abundance_matrix)
  gc(verbose = FALSE)
}

all_species <- sort(all_species)

cat("Global species vocabulary size:", length(all_species), "\n")
writeLines(all_species, "species_vocab.txt")

# -----------------------------
# PASS 2: Align each study to global vocabulary
# -----------------------------


for (i in seq_along(target_studies)) {
  study <- target_studies[i]
  
  se_list <- tryCatch({
    curatedMetagenomicData(study, dryrun = FALSE, counts = FALSE)
  }, error = function(e) NULL)
  
  if (is.null(se_list) || length(se_list) == 0) next
  se <- se_list[[1]]

  # Filter only stool
  metadata <- as.data.frame(colData(se))
  if (!"body_site" %in% colnames(metadata)) next

  # Convert row names to an explicit column so pull(sample_id) works
  stool_metadata <- metadata %>% 
    tibble::rownames_to_column("sample_id") %>% 
    filter(body_site == "stool") #%>% 

    # COMMENTED OUT TO RETAIN ALL SAMPLES FOR PRE-TRAINING, NOT JUST ONE PER SUBJECT
    # KEEP ONLY THE FIRST SAMPLE PER UNIQUE SUBJECT ID
    # distinct(subject_id, .keep_all = TRUE)
    
  stool_ids <- stool_metadata %>% pull(sample_id)
    
  if (length(stool_ids) == 0) next

  print("processing study, stool samples found: ")
  print(length(stool_ids))
  # Subset and extract species
# --- FIX IS HERE: Subset using colnames() instead of a missing metadata column ---
  se_stool <- se[, colnames(se) %in% stool_ids, drop = FALSE]

  if (ncol(se_stool) == 0) next

  abundance_matrix <- assay(se_stool)
  species_rows <- grep("s__", rownames(abundance_matrix), value = TRUE)
  species_rows <- species_rows[!grepl("t__", species_rows)]

  if (length(species_rows) == 0) next
  matrix_species <- abundance_matrix[species_rows, , drop = FALSE]

# Keep full clade strings as column keys
rownames(matrix_species) <- species_rows

# Transpose: rows = samples, columns = species
pivoted_matrix <- t(matrix_species)
df_pivoted <- as.data.frame(pivoted_matrix)

# Move Sample IDs into explicit column
df_pivoted <- tibble::rownames_to_column(df_pivoted, "Sample_ID")

# Add Study ID for debugging/batch analysis
df_pivoted$Study_ID <- study

# Add missing species columns as zero
missing_species <- setdiff(all_species, colnames(df_pivoted))

if (length(missing_species) > 0) {
  df_pivoted[missing_species] <- 0
}

# Reorder columns to fixed global order
df_pivoted <- df_pivoted[, c("Study_ID", "Sample_ID", all_species), drop = FALSE]

# Write safely
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
  
  rm(se, se_list, se_stool, abundance_matrix, matrix_species, df_pivoted); gc(verbose = FALSE)
}
cat("Pre-training matrix generation complete!\n")