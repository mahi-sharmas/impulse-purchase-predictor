# =============================================================================
# LAB 8: Classification Models
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab3_features.R (train_data.rds, test_data.rds — UNSCALED)
#             lab7_regression.R (logistic_metrics.rds, roc_logistic.rds)
# NOTE: Decision Tree and Random Forest are scale-invariant — use unscaled data
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
# rpart is excluded — it's a caret dependency and is already loaded in session
pkgs <- c("rpart.plot", "randomForest", "caret", "pROC", "ggplot2", "smotefamily")
new_pkgs <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new_pkgs)) install.packages(new_pkgs, quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(rpart.plot)
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)
library(smotefamily)

# ── Load unscaled data ────────────────────────────────────────────────────────
train <- readRDS("/content/data/engineered/train_data.rds")
test  <- readRDS("/content/data/engineered/test_data.rds")

# Fix column names with spaces (e.g. "payment_method.Credit Card" → valid R name)
names(train) <- make.names(names(train))
names(test)  <- make.names(names(test))

target_col <- "impulse_buyer"
spend_col  <- "total_purchase_amount"  # excluded from features (defines target)

cat("Train:", nrow(train), "rows | Test:", nrow(test), "rows\n")
cat("Class distribution (train):\n")
print(round(prop.table(table(train[[target_col]])) * 100, 1))

# =============================================================================
# STEP 1: SMOTE — Handle Class Imbalance (applied to train only)
# =============================================================================
class_ratio  <- prop.table(table(train[[target_col]]))
majority_pct <- max(class_ratio) * 100

if (majority_pct > 65) {
  cat("\n⚠ Imbalance detected — applying SMOTE to training set\n")

  # SMOTE requires numeric matrix — convert factors to numeric
  train_x <- train %>% select(-all_of(target_col))
  train_x_num <- as.data.frame(lapply(train_x, function(col) {
    if (is.factor(col)) as.numeric(col) else col
  }))
  train_y <- train[[target_col]]

  smote_result   <- SMOTE(X = train_x_num, target = as.numeric(train_y) - 1, K = 5)
  train_balanced <- smote_result$data
  names(train_balanced)[ncol(train_balanced)] <- target_col
  train_balanced[[target_col]] <- factor(
    ifelse(train_balanced[[target_col]] == 0, levels(train_y)[1], levels(train_y)[2]),
    levels = levels(train_y)
  )
  cat("After SMOTE:\n")
  print(round(prop.table(table(train_balanced[[target_col]])) * 100, 1))
} else {
  cat("✓ Classes balanced — no SMOTE needed\n")
  train_balanced <- train
}

# Feature set: exclude target and spend_col (spend_col was used to create target)
model_features <- setdiff(names(train_balanced), c(target_col, spend_col))
model_formula  <- as.formula(paste(target_col, "~",
                                    paste(model_features, collapse = " + ")))

set.seed(42)
cv_ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)

# =============================================================================
# STEP 2: DECISION TREE
# =============================================================================
cat("\n========== DECISION TREE ==========\n")
set.seed(42)

dt_model <- train(
  model_formula,
  data       = train_balanced,
  method     = "rpart",
  trControl  = cv_ctrl,
  metric     = "ROC",
  tuneLength = 10
)
cat("Best cp:", dt_model$bestTune$cp, "\n")

# Visualize
png("/content/outputs/plots/decision_tree.png", width=1200, height=800, res=120)
rpart.plot(dt_model$finalModel, type=4, extra=104,
           main = "Decision Tree — Impulse Buyer Prediction")
dev.off()
cat("✓ Decision tree plot saved\n")

# Evaluate
dt_probs <- predict(dt_model, newdata = test, type = "prob")[, "Yes"]
dt_class <- predict(dt_model, newdata = test)
dt_cm    <- confusionMatrix(dt_class, test[[target_col]], positive = "Yes")
dt_roc   <- roc(test[[target_col]], dt_probs, quiet = TRUE)
dt_auc   <- auc(dt_roc)

cat(sprintf("\nDecision Tree  → Accuracy: %.4f | AUC: %.4f | F1: %.4f\n",
            dt_cm$overall["Accuracy"], dt_auc, dt_cm$byClass["F1"]))

saveRDS(dt_model, "/content/outputs/models/decision_tree.rds")
saveRDS(dt_roc,   "/content/outputs/models/roc_dt.rds")

# =============================================================================
# STEP 3: RANDOM FOREST
# =============================================================================
cat("\n========== RANDOM FOREST ==========\n")
set.seed(42)

rf_grid <- expand.grid(
  mtry = c(2, 3, floor(sqrt(length(model_features))), 5)
)

rf_model <- train(
  model_formula,
  data      = train_balanced,
  method    = "rf",
  trControl = cv_ctrl,
  metric    = "ROC",
  tuneGrid  = rf_grid,
  ntree     = 200
)
cat("Best mtry:", rf_model$bestTune$mtry, "\n")

# Evaluate
rf_probs <- predict(rf_model, newdata = test, type = "prob")[, "Yes"]
rf_class <- predict(rf_model, newdata = test)
rf_cm    <- confusionMatrix(rf_class, test[[target_col]], positive = "Yes")
rf_roc   <- roc(test[[target_col]], rf_probs, quiet = TRUE)
rf_auc   <- auc(rf_roc)

cat(sprintf("Random Forest  → Accuracy: %.4f | AUC: %.4f | F1: %.4f\n",
            rf_cm$overall["Accuracy"], rf_auc, rf_cm$byClass["F1"]))

saveRDS(rf_model, "/content/outputs/models/random_forest.rds")
saveRDS(rf_roc,   "/content/outputs/models/roc_rf.rds")

# =============================================================================
# STEP 4: FEATURE IMPORTANCE (Random Forest)
# =============================================================================
cat("\n========== FEATURE IMPORTANCE ==========\n")

imp_df     <- as.data.frame(importance(rf_model$finalModel))
imp_df$Feature <- rownames(imp_df)
imp_col    <- if ("MeanDecreaseGini" %in% names(imp_df)) "MeanDecreaseGini" else names(imp_df)[1]
top10      <- head(imp_df[order(imp_df[[imp_col]], decreasing=TRUE), ], 10)

p_imp <- ggplot(top10, aes(x = reorder(Feature, .data[[imp_col]]),
                            y = .data[[imp_col]])) +
  geom_bar(stat = "identity", fill = "#4E9AF1") +
  coord_flip() +
  labs(title = "Top 10 Feature Importances (Random Forest)",
       x = "", y = "Mean Decrease Gini") +
  theme_minimal()

print(p_imp)
ggsave("/content/outputs/plots/feature_importance.png",
       plot = p_imp, width = 8, height = 5, dpi = 150)
cat("✓ Feature importance plot saved\n")

# =============================================================================
# STEP 5: MODEL COMPARISON TABLE + ROC OVERLAY
# =============================================================================
cat("\n========== MODEL COMPARISON ==========\n")

logistic_metrics <- tryCatch(
  readRDS("/content/outputs/models/logistic_metrics.rds"),
  error = function(e) data.frame(Model="Logistic Regression",
                                  Accuracy=NA, AUC=NA, Precision=NA, Recall=NA, F1=NA)
)
roc_logistic <- tryCatch(
  readRDS("/content/outputs/models/roc_logistic.rds"),
  error = function(e) NULL
)

comparison <- rbind(
  logistic_metrics,
  data.frame(Model="Decision Tree",
             Accuracy  = round(dt_cm$overall["Accuracy"],  4),
             AUC       = round(dt_auc,                     4),
             Precision = round(dt_cm$byClass["Precision"], 4),
             Recall    = round(dt_cm$byClass["Recall"],    4),
             F1        = round(dt_cm$byClass["F1"],        4)),
  data.frame(Model="Random Forest",
             Accuracy  = round(rf_cm$overall["Accuracy"],  4),
             AUC       = round(rf_auc,                     4),
             Precision = round(rf_cm$byClass["Precision"], 4),
             Recall    = round(rf_cm$byClass["Recall"],    4),
             F1        = round(rf_cm$byClass["F1"],        4))
)
rownames(comparison) <- NULL
print(comparison)

# Overlapping ROC curves
png("/content/outputs/plots/roc_comparison.png", width=800, height=700, res=120)
plot(dt_roc, col="#E07B54", lwd=2,
     main="ROC Curves — Model Comparison")
lines(rf_roc, col="#4E9AF1", lwd=2)
if (!is.null(roc_logistic))
  lines(roc_logistic, col="#2CA02C", lwd=2)
legend("bottomright",
       legend = c(paste0("Decision Tree (AUC=",    round(dt_auc,  3), ")"),
                  paste0("Random Forest (AUC=",    round(rf_auc,  3), ")"),
                  if (!is.null(roc_logistic))
                    paste0("Logistic Reg. (AUC=",
                           round(auc(roc_logistic), 3), ")")),
       col = c("#E07B54","#4E9AF1","#2CA02C"), lwd = 2)
abline(a=0, b=1, col="gray", lty=2)
dev.off()
cat("✓ ROC comparison plot saved\n")

# Save best model
best_idx   <- which.max(comparison$AUC)
best_label <- comparison$Model[best_idx]
best_model <- switch(best_label,
  "Logistic Regression" = readRDS("/content/outputs/models/logistic_model.rds"),
  "Decision Tree"       = dt_model,
  "Random Forest"       = rf_model
)
saveRDS(best_model, "/content/outputs/models/best_classifier.rds")
cat("\n✓ Best model:", best_label,
    "| AUC =", comparison$AUC[best_idx], "\n")
cat("✓ Saved as best_classifier.rds\n")

cat("\n=== LAB 8 COMPLETE ===\n")
cat("Next: Run lab9_clustering.R\n")
