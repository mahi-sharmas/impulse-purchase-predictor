# =============================================================================
# LAB 1: Data Collection and Import
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# HOW TO USE IN COLAB:
#   1. Upload your ZIP file using the Colab file panel (left sidebar)
#   2. Run this script cell by cell
# =============================================================================

# ── Install packages (run once per Colab session) ────────────────────────────
install.packages(c("readr"),
                 quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(readr)

# ── Step 1: Create project directory structure ────────────────────────────────
dirs <- c(
  "/content/data/raw",
  "/content/data/cleaned",
  "/content/data/engineered",
  "/content/outputs/plots",
  "/content/outputs/models",
  "/content/dashboard"
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))
cat("✓ Project directories created\n")

# ── Step 2: Extract ZIP and locate CSV ───────────────────────────────────────
# ADJUST: change "archive.zip" to your actual uploaded ZIP filename
zip_path <- "/content/archive.zip"

if (file.exists(zip_path)) {
  unzip(zip_path, exdir = "/content/data/raw")
  cat("✓ ZIP extracted to /content/data/raw/\n")
} else {
  stop("ZIP file not found. Please upload it to /content/ first.")
}

# Find the CSV inside the extracted folder
csv_files <- list.files("/content/data/raw", pattern = "\\.csv$",
                        recursive = TRUE, full.names = TRUE)
cat("CSV files found:\n")
print(csv_files)

# Use the first CSV found (adjust index if multiple CSVs exist)
csv_path <- csv_files[1]

# ── Step 3: Import CSV (primary method) ──────────────────────────────────────
df_raw <- read_csv(csv_path, show_col_types = FALSE)
cat("\n✓ Dataset loaded successfully\n")

# ── Step 4: Inspect the dataset ──────────────────────────────────────────────
cat("\n--- Dataset Dimensions ---\n")
print(dim(df_raw))             # rows x columns

cat("\n--- Column Names ---\n")
print(names(df_raw))

cat("\n--- Data Types & Structure ---\n")
str(df_raw)

cat("\n--- First 10 Rows ---\n")
print(head(df_raw, 10))

cat("\n--- Summary Statistics ---\n")
print(summary(df_raw))

# ── Step 5: Save raw snapshot (never overwrite this) ─────────────────────────
saveRDS(df_raw, "/content/data/raw/raw_data.rds")
cat("\n✓ Raw data saved to /content/data/raw/raw_data.rds\n")



cat("\n=== LAB 1 COMPLETE ===\n")
cat("Next: Run lab2_cleaning.R\n")
