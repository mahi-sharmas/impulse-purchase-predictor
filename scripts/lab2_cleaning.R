# =============================================================================
# LAB 2: Data Cleaning Techniques
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab1_import.R (raw_data.rds must exist)
# DATASET COLUMNS (actual):
#   Customer ID, Purchase Date, Product Category, Product Price, Quantity,
#   Total Purchase Amount, Payment Method, Customer Age, Returns,
#   Customer Name, Age (duplicate), Gender, Churn
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("dplyr", "tidyr", "lubridate", "janitor", "naniar"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(dplyr)
library(tidyr)
library(lubridate)
library(janitor)
library(naniar)

# ── Load raw data ─────────────────────────────────────────────────────────────
df <- readRDS("/content/data/raw/raw_data.rds")
cat("Loaded dataset:", nrow(df), "rows,", ncol(df), "columns\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Clean Column Names → snake_case
# Converts "Total Purchase Amount" → total_purchase_amount, etc.
# ─────────────────────────────────────────────────────────────────────────────
df <- clean_names(df)
cat("\n✓ Column names cleaned to snake_case:\n")
print(names(df))
# Result: customer_id, purchase_date, product_category, product_price, quantity,
#         total_purchase_amount, payment_method, customer_age, returns,
#         customer_name, age, gender, churn

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Engineer the Target Variable — impulse_buyer
# Definition: customers whose total_purchase_amount is above the 60th percentile
# are labeled impulse buyers (top 40% spenders = high-value impulsive purchasing)
# ─────────────────────────────────────────────────────────────────────────────
threshold <- quantile(df$total_purchase_amount, 0.60, na.rm = TRUE)
df$impulse_buyer <- factor(
  ifelse(df$total_purchase_amount > threshold, "Yes", "No"),
  levels = c("No", "Yes")
)
cat("\n✓ Target variable 'impulse_buyer' created (threshold:", threshold, ")\n")
cat("Class distribution:\n")
print(table(df$impulse_buyer))
print(round(prop.table(table(df$impulse_buyer)) * 100, 1))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Remove Unnecessary Columns
# ─────────────────────────────────────────────────────────────────────────────
# Drop: ID cols (no predictive value), customer_name (PII),
#       age (exact duplicate of customer_age), churn (unrelated target)
df <- df %>% select(-customer_id, -customer_name, -age, -churn)
cat("\n✓ Dropped: customer_id, customer_name, age, churn\n")
cat("Remaining columns:", paste(names(df), collapse = ", "), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Missing Value Audit
# ─────────────────────────────────────────────────────────────────────────────
na_counts <- colSums(is.na(df))
cat("\n--- Missing Value Report ---\n")
print(na_counts[na_counts > 0])
# Expected: returns has ~47,596 NAs (~19%)

# Visual NA map — sample 5000 rows (vis_miss can't handle 250k rows)
vis_miss(dplyr::slice_sample(df, n = 5000))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Handle Missing Values
# ─────────────────────────────────────────────────────────────────────────────

# returns: NAs mean the return status was not recorded → treat as no return (0)
df$returns <- replace_na(df$returns, 0)
cat("\n✓ 'returns' NAs imputed with 0 (not returned)\n")

# Confirm no remaining NAs
cat("Total NAs remaining:", sum(is.na(df)), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Data Type Conversion
# ─────────────────────────────────────────────────────────────────────────────

# purchase_date is already POSIXct — keep as-is for Lab 3 feature extraction
# Convert categorical columns to factor
df <- df %>%
  mutate(
    product_category = as.factor(product_category),
    payment_method   = as.factor(payment_method),
    gender           = as.factor(gender),
    returns          = as.integer(returns)   # ensure integer (0/1)
  )

cat("\n✓ Type conversions applied\n")
cat("Updated types:\n")
print(sapply(df, class))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Remove Duplicate Rows
# ─────────────────────────────────────────────────────────────────────────────
rows_before <- nrow(df)
df <- df %>% distinct()
cat("\nDuplicate rows removed:", rows_before - nrow(df), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Final Summary & Save
# ─────────────────────────────────────────────────────────────────────────────
cat("\n--- Final Cleaned Dataset ---\n")
cat("Rows:", nrow(df), "| Columns:", ncol(df), "\n")
cat("Target distribution:\n")
print(table(df$impulse_buyer))

write.csv(df, "/content/data/cleaned/cleaned_data.csv", row.names = FALSE)
saveRDS(df,   "/content/data/cleaned/cleaned_data.rds")
cat("\n✓ Cleaned data saved to /content/data/cleaned/\n")

cat("\n=== LAB 2 COMPLETE ===\n")
cat("Next: Run lab3_features.R\n")
