# =============================================================================
# LAB 7: Regression Models
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab3_features.R (train_scaled.rds, test_scaled.rds)
#
# Part A — Linear Regression: predict total_purchase_amount (supporting analysis
#           to understand what drives spending behavior)
# Part B — Logistic Regression: predict impulse_buyer (main classification task)
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("caret", "pROC", "ggplot2"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(caret)
library(pROC)
library(ggplot2)

# ── Load scaled train/test splits ─────────────────────────────────────────────
train <- readRDS("/content/data/engineered/train_scaled.rds")
test  <- readRDS("/content/data/engineered/test_scaled.rds")

# Fix column names with spaces (e.g. "payment_method.Credit Card" → valid R name)
names(train) <- make.names(names(train))
names(test)  <- make.names(names(test))

target_col <- "impulse_buyer"
spend_col  <- "total_purchase_amount"

cat("Train:", nrow(train), "rows | Test:", nrow(test), "rows\n")

# =============================================================================
# PART A: LINEAR REGRESSION — Supporting Analysis (Spending Prediction)
# Linear regression was used as a supporting analysis to understand spending
# behavior, while classification models are used for the main prediction task.
# =============================================================================
cat("\n========== PART A: LINEAR REGRESSION ==========\n")

# All features except the classification target
lm_features <- setdiff(names(train), c(target_col))
lm_formula  <- as.formula(paste(spend_col, "~",
                                 paste(setdiff(lm_features, spend_col), collapse = " + ")))

lm_model <- lm(lm_formula, data = train)

lm_preds   <- predict(lm_model, newdata = test)
lm_metrics <- postResample(pred = lm_preds, obs = test[[spend_col]])

cat("\n--- Linear Regression Metrics (Test Set) ---\n")
cat(sprintf("  RMSE      : %.3f\n", lm_metrics["RMSE"]))
cat(sprintf("  MAE       : %.3f\n", mean(abs(lm_preds - test[[spend_col]]))))
cat(sprintf("  R-squared : %.4f\n", lm_metrics["Rsquared"]))

# Actual vs. Predicted plot
p_lm <- ggplot(data.frame(actual = test[[spend_col]], predicted = lm_preds),
               aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.3, color = "#4E9AF1", size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title    = "Linear Regression: Actual vs. Predicted Spending",
       subtitle = paste0("R² = ", round(lm_metrics["Rsquared"], 3),
                         "  |  RMSE = ", round(lm_metrics["RMSE"], 2)),
       x = "Actual Total Purchase Amount",
       y = "Predicted Total Purchase Amount") +
  theme_minimal()

print(p_lm)
ggsave("/content/outputs/plots/lm_actual_vs_predicted.png",
       plot = p_lm, width = 7, height = 5, dpi = 150)
cat("✓ Plot saved: lm_actual_vs_predicted.png\n")

saveRDS(lm_model, "/content/outputs/models/linear_model.rds")
cat("✓ Linear model saved\n")

# =============================================================================
# PART B: LOGISTIC REGRESSION — Main Classification Model
# =============================================================================
cat("\n========== PART B: LOGISTIC REGRESSION ==========\n")

# Build formula (exclude spend_col to avoid leakage — it defines the target)
glm_features <- setdiff(names(train), c(target_col, spend_col))
glm_formula  <- as.formula(paste(target_col, "~",
                                  paste(glm_features, collapse = " + ")))

# 5-fold cross-validation
set.seed(42)
cv_ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

glm_model <- train(
  glm_formula,
  data      = train,
  method    = "glm",
  family    = "binomial",
  trControl = cv_ctrl,
  metric    = "ROC"
)

cat("\n--- 5-Fold CV Results ---\n")
print(glm_model$results[, c("ROC", "Sens", "Spec")])

# Predict on test set
glm_probs <- predict(glm_model, newdata = test, type = "prob")[, "Yes"]
glm_class <- predict(glm_model, newdata = test)

# Confusion matrix
cat("\n--- Confusion Matrix (Test Set) ---\n")
cm <- confusionMatrix(glm_class, test[[target_col]], positive = "Yes")
print(cm)

# ROC curve & AUC
roc_obj <- roc(test[[target_col]], glm_probs, quiet = TRUE)
auc_val <- auc(roc_obj)
cat(sprintf("\nAUC (Test Set): %.4f\n", auc_val))

png("/content/outputs/plots/roc_logistic.png", width = 700, height = 600, res = 120)
plot(roc_obj,
     main      = paste0("Logistic Regression — ROC Curve (AUC = ", round(auc_val, 3), ")"),
     col       = "#4E9AF1", lwd = 2,
     print.auc = TRUE)
abline(a = 0, b = 1, col = "gray", lty = 2)
dev.off()
cat("✓ ROC curve saved: roc_logistic.png\n")

# Save summary metrics (used in Lab 8 comparison table)
logistic_metrics <- data.frame(
  Model     = "Logistic Regression",
  Accuracy  = round(cm$overall["Accuracy"],   4),
  AUC       = round(auc_val,                  4),
  Precision = round(cm$byClass["Precision"],  4),
  Recall    = round(cm$byClass["Recall"],     4),
  F1        = round(cm$byClass["F1"],         4)
)
cat("\n--- Logistic Regression Summary ---\n")
print(logistic_metrics)

saveRDS(glm_model,        "/content/outputs/models/logistic_model.rds")
saveRDS(logistic_metrics, "/content/outputs/models/logistic_metrics.rds")
saveRDS(roc_obj,          "/content/outputs/models/roc_logistic.rds")
cat("✓ Logistic model saved\n")

cat("\n=== LAB 7 COMPLETE ===\n")
cat("Next: Run lab8_classification.R\n")
