#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(dplyr)
})

all_resources <- curatedMetagenomicData(".relative_abundance", dryrun = TRUE)
target_studies <- all_resources[grepl("\\.relative_abundance", all_resources)]

cat("Listing disease counts for stool samples per study...\n\n")

global_counts <- list()

for (study in target_studies) {
  se_list <- tryCatch({
    curatedMetagenomicData(study, dryrun = FALSE, counts = FALSE)
  }, error = function(e) NULL)

  if (is.null(se_list) || length(se_list) == 0) next

  se <- se_list[[1]]
  metadata <- as.data.frame(colData(se))

  if (!"body_site" %in% colnames(metadata)) next
  if (!"disease" %in% colnames(metadata)) next

  stool_metadata <- metadata %>%
    tibble::rownames_to_column("sample_id") %>%
    filter(body_site == "stool") %>%
    filter(!is.na(disease)) %>%
    filter(disease != "")

  if (nrow(stool_metadata) == 0) next

  disease_counts <- stool_metadata %>%
    count(disease, sort = TRUE)

  cat("========================================\n")
  cat("Study:", study, "\n")
  cat("Total stool samples with disease label:", nrow(stool_metadata), "\n")
  cat("----------------------------------------\n")
  print(disease_counts, n = Inf)
  cat("\n")

  global_counts[[length(global_counts) + 1]] <- disease_counts %>%
    mutate(study = study)

  rm(se, se_list, metadata, stool_metadata, disease_counts)
  gc(verbose = FALSE)
}

cat("========================================\n")
cat("GLOBAL DISEASE COUNTS ACROSS ALL STUDIES\n")
cat("========================================\n")

if (length(global_counts) > 0) {
  global_disease_counts <- bind_rows(global_counts) %>%
    group_by(disease) %>%
    summarise(
      total_samples = sum(n),
      num_studies = n_distinct(study),
      .groups = "drop"
    ) %>%
    arrange(desc(total_samples))

  print(global_disease_counts, n = Inf)

  cat("\nLargest disease class:\n")
  print(global_disease_counts[1, ])
} else {
  cat("No stool disease labels found.\n")
}