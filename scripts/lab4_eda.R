# =============================================================================
# LAB 4: Exploratory Data Analysis (EDA)
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab2_cleaning.R (cleaned_data.rds)
# Run EDA on cleaned data (before train/test split)
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("dplyr", "tidyr"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(dplyr)
library(tidyr)

# ── Load cleaned data ─────────────────────────────────────────────────────────
df <- readRDS("/content/data/cleaned/cleaned_data.rds")
cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n")
cat("Columns:", paste(names(df), collapse = ", "), "\n")

target_col <- "impulse_buyer"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────
cat("\n========== SUMMARY STATISTICS ==========\n")
print(summary(df))

# Target class distribution
cat("\n--- Target: impulse_buyer ---\n")
print(table(df$impulse_buyer))
print(round(prop.table(table(df$impulse_buyer)) * 100, 1))

majority_pct <- max(prop.table(table(df$impulse_buyer))) * 100
if (majority_pct > 65) {
  cat("\n⚠ CLASS IMBALANCE:", round(majority_pct, 1), "% majority — consider SMOTE in Lab 8\n")
} else {
  cat("✓ Classes are reasonably balanced\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Univariate Analysis
# ─────────────────────────────────────────────────────────────────────────────
cat("\n========== UNIVARIATE ANALYSIS ==========\n")

# Numeric features
numeric_cols <- c("product_price", "quantity", "total_purchase_amount", "customer_age", "returns")
cat("\n--- Numeric Features ---\n")
for (col in numeric_cols) {
  if (col %in% names(df)) {
    x <- df[[col]]
    cat(sprintf("%-25s  mean=%7.2f  median=%7.2f  sd=%7.2f  min=%5.0f  max=%5.0f\n",
                col, mean(x, na.rm=T), median(x, na.rm=T),
                sd(x, na.rm=T), min(x, na.rm=T), max(x, na.rm=T)))
  }
}

# Categorical features
cat("\n--- Categorical Features ---\n")
cat("\nProduct Category:\n");  print(sort(table(df$product_category), decreasing=TRUE))
cat("\nPayment Method:\n");    print(sort(table(df$payment_method),   decreasing=TRUE))
cat("\nGender:\n");            print(table(df$gender))
cat("\nReturns (0=No, 1=Yes):\n"); print(table(df$returns))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Multivariate Analysis — Compare Impulse vs. Non-Impulse Buyers
# ─────────────────────────────────────────────────────────────────────────────
cat("\n========== MULTIVARIATE ANALYSIS ==========\n")
cat("--- Mean of Numeric Features by impulse_buyer ---\n")

group_means <- df %>%
  group_by(impulse_buyer) %>%
  summarise(
    avg_product_price        = round(mean(product_price,          na.rm=T), 2),
    avg_quantity             = round(mean(quantity,               na.rm=T), 2),
    avg_total_purchase_amt   = round(mean(total_purchase_amount,  na.rm=T), 2),
    avg_customer_age         = round(mean(customer_age,           na.rm=T), 2),
    return_rate_pct          = round(mean(returns,                na.rm=T) * 100, 1),
    .groups = "drop"
  )
print(as.data.frame(group_means))

# Return rate by impulse class
cat("\n--- Return Rate by Impulse Buyer Class ---\n")
return_by_class <- df %>%
  group_by(impulse_buyer) %>%
  summarise(return_rate = round(mean(returns, na.rm=T) * 100, 1), .groups="drop")
print(as.data.frame(return_by_class))

# Impulse rate by product category
cat("\n--- Impulse Buyer Rate by Product Category ---\n")
impulse_by_cat <- df %>%
  group_by(product_category) %>%
  summarise(
    impulse_rate = round(mean(impulse_buyer == "Yes") * 100, 1),
    n = n(),
    .groups = "drop"
  ) %>% arrange(desc(impulse_rate))
print(as.data.frame(impulse_by_cat))

# Impulse rate by payment method
cat("\n--- Impulse Buyer Rate by Payment Method ---\n")
impulse_by_pay <- df %>%
  group_by(payment_method) %>%
  summarise(
    impulse_rate = round(mean(impulse_buyer == "Yes") * 100, 1),
    n = n(),
    .groups = "drop"
  ) %>% arrange(desc(impulse_rate))
print(as.data.frame(impulse_by_pay))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Correlation Analysis (numeric features)
# ─────────────────────────────────────────────────────────────────────────────
cat("\n========== CORRELATION ANALYSIS ==========\n")

num_df     <- df[, numeric_cols[numeric_cols %in% names(df)]]
cor_matrix <- cor(num_df, use = "complete.obs")
cat("--- Correlation Matrix ---\n")
print(round(cor_matrix, 3))

# Flag highly correlated pairs
high_cor <- which(abs(cor_matrix) > 0.8 & upper.tri(cor_matrix), arr.ind = TRUE)
if (nrow(high_cor) > 0) {
  cat("\n⚠ Highly correlated pairs (|r| > 0.8):\n")
  for (i in seq_len(nrow(high_cor))) {
    r  <- cor_matrix[high_cor[i,1], high_cor[i,2]]
    c1 <- rownames(cor_matrix)[high_cor[i,1]]
    c2 <- colnames(cor_matrix)[high_cor[i,2]]
    cat(sprintf("  %s  <-->  %s  :  r = %.3f\n", c1, c2, r))
  }
} else {
  cat("✓ No highly correlated pairs found\n")
}

cat("\n=== LAB 4 COMPLETE ===\n")
cat("Next: Run lab5_visualization.R\n")
