# =============================================================================
# LAB 3: Feature Engineering & Transformation
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab2_cleaning.R (cleaned_data.rds)
# COLUMNS AVAILABLE (after Lab 2 cleaning):
#   purchase_date, product_category, product_price, quantity,
#   total_purchase_amount, payment_method, customer_age, returns,
#   gender, impulse_buyer (target)
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("dplyr", "lubridate", "caret"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(dplyr)
library(lubridate)
library(caret)

# ── Load cleaned data ─────────────────────────────────────────────────────────
df <- readRDS("/content/data/cleaned/cleaned_data.rds")
cat("Loaded cleaned dataset:", nrow(df), "rows,", ncol(df), "columns\n")

target_col <- "impulse_buyer"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Date Feature Extraction from purchase_date
# ─────────────────────────────────────────────────────────────────────────────
df <- df %>%
  mutate(
    day_of_week  = wday(purchase_date, label = TRUE, abbr = TRUE),   # Mon–Sun
    is_weekend   = as.integer(wday(purchase_date) %in% c(1, 7)),     # 1=Sun, 7=Sat
    hour_of_day  = hour(purchase_date),                               # 0–23
    month        = month(purchase_date, label = TRUE, abbr = TRUE)    # Jan–Dec
  )

# Drop purchase_date — raw datetime not usable in ML models
df <- df %>% select(-purchase_date)

cat("✓ Date features extracted: day_of_week, is_weekend, hour_of_day, month\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Behavioral Feature Creation
# ─────────────────────────────────────────────────────────────────────────────
df <- df %>%
  mutate(
    # Spending per unit item (higher = possibly premium/impulse purchase)
    price_per_item       = total_purchase_amount / quantity,

    # Flag high-quantity purchases (buying 3+ items = bulk/impulse behavior)
    high_quantity_flag   = as.integer(quantity >= 3),

    # Flag customers with above-median product price (premium buyer signal)
    high_price_flag      = as.integer(product_price > median(product_price, na.rm = TRUE))
  )

cat("✓ Behavioral features created: price_per_item, high_quantity_flag, high_price_flag\n")
cat("Columns now:", ncol(df), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

# 3a. Remove near-zero variance features
feature_df <- df %>% select(-all_of(target_col))
nzv_idx    <- nearZeroVar(feature_df)
if (length(nzv_idx) > 0) {
  nzv_names <- names(feature_df)[nzv_idx]
  cat("Removing near-zero variance features:", paste(nzv_names, collapse = ", "), "\n")
  df <- df %>% select(-all_of(nzv_names))
} else {
  cat("✓ No near-zero variance features found\n")
}

# 3b. Remove highly correlated numeric features (|r| > 0.85)
numeric_cols <- names(df)[sapply(df, is.numeric) & names(df) != target_col]
if (length(numeric_cols) > 1) {
  cor_mat      <- cor(df[, numeric_cols], use = "complete.obs")
  high_cor_idx <- findCorrelation(cor_mat, cutoff = 0.85, verbose = FALSE)
  if (length(high_cor_idx) > 0) {
    high_cor_names <- numeric_cols[high_cor_idx]
    cat("Removing highly correlated features (|r|>0.85):",
        paste(high_cor_names, collapse = ", "), "\n")
    df <- df %>% select(-all_of(high_cor_names))
  } else {
    cat("✓ No highly correlated pairs found\n")
  }
}

cat("Columns after feature selection:", ncol(df), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Encode Categorical Variables (one-hot)
# ─────────────────────────────────────────────────────────────────────────────
target_vec  <- df[[target_col]]
df_features <- df %>% select(-all_of(target_col))

# Apply one-hot encoding to all factor columns
dummies     <- dummyVars("~ .", data = df_features, fullRank = TRUE)
df_encoded  <- as.data.frame(predict(dummies, newdata = df_features))
df_encoded[[target_col]] <- target_vec

cat("✓ One-hot encoding applied. Columns after encoding:", ncol(df_encoded), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Train / Test Split (70/30, stratified by target)
# ─────────────────────────────────────────────────────────────────────────────
set.seed(42)
train_idx  <- createDataPartition(df_encoded[[target_col]], p = 0.70, list = FALSE)
train_data <- df_encoded[ train_idx, ]
test_data  <- df_encoded[-train_idx, ]

cat("\n--- Train/Test Split ---\n")
cat("Train:", nrow(train_data), "rows | Test:", nrow(test_data), "rows\n")
cat("Train class distribution:\n")
print(round(prop.table(table(train_data[[target_col]])) * 100, 1))
cat("Test class distribution:\n")
print(round(prop.table(table(test_data[[target_col]])) * 100, 1))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Feature Scaling (for Logistic Regression and Clustering only)
# Fit scaler on TRAIN set only — then apply the same scaler to TEST set
# ─────────────────────────────────────────────────────────────────────────────
numeric_features <- names(train_data)[
  sapply(train_data, is.numeric) & names(train_data) != target_col
]

preproc      <- preProcess(train_data[, numeric_features], method = c("center", "scale"))
train_scaled <- train_data
test_scaled  <- test_data
train_scaled[, numeric_features] <- predict(preproc, train_data[, numeric_features])
test_scaled[, numeric_features]  <- predict(preproc, test_data[, numeric_features])

cat("✓ Scaling applied (fitted on train set only)\n")

# ─────────────────────────────────────────────────────────────────────────────
# Save all outputs
# ─────────────────────────────────────────────────────────────────────────────
saveRDS(train_data,   "/content/data/engineered/train_data.rds")
saveRDS(test_data,    "/content/data/engineered/test_data.rds")
saveRDS(train_scaled, "/content/data/engineered/train_scaled.rds")
saveRDS(test_scaled,  "/content/data/engineered/test_scaled.rds")
saveRDS(preproc,      "/content/data/engineered/preproc_scaler.rds")

cat("\n✓ All splits saved to /content/data/engineered/\n")

cat("\n=== LAB 3 COMPLETE ===\n")
cat("Next: Run lab4_eda.R\n")
