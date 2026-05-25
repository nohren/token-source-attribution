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

for (i in seq_along(target_studies)) {
  study <- target_studies[i]
  
  se_list <- tryCatch({
    curatedMetagenomicData(study, dryrun = FALSE, counts = FALSE)
  }, error = function(e) next)
  
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

  abundance_matrix <- assay(se_stool)
  species_rows <- grep("s__", rownames(abundance_matrix), value = TRUE)
  species_rows <- species_rows[!grepl("t__", species_rows)]

  if (length(species_rows) == 0) next
  matrix_species <- abundance_matrix[species_rows, , drop = FALSE]
  rownames(matrix_species) <- sub(".*s__", "", rownames(matrix_species))
  
  # 5. Transpose to ML format: Rows = Samples, Columns = Species
  pivoted_matrix <- t(matrix_species)
  
  # Convert explicitly to dataframe, preserving row names cleanly
  df_pivoted <- as.data.frame(pivoted_matrix)

  # print("pivoted matrix: ")
  # print(df_pivoted)
  
  # Move the Sample IDs out of the row name space into a dedicated first column
  df_pivoted <- tibble::rownames_to_column(df_pivoted, "Sample_ID")
  
  # 6. Stream directly to disk
  if (is_first_write) {
    # Write fresh file with headers
    write.table(df_pivoted, file = output_file, sep = "\t", 
                row.names = FALSE, col.names = TRUE, quote = FALSE)
    is_first_write <- FALSE
  } else {
    # Append numerical rows to existing file WITHOUT headers
    write.table(df_pivoted, file = output_file, sep = "\t", 
                append = TRUE, row.names = FALSE, col.names = FALSE, quote = FALSE)
  }
  
  rm(se, se_list, se_stool, abundance_matrix, matrix_species, df_pivoted); gc(verbose = FALSE)
}
cat("Pre-training matrix generation complete!\n")