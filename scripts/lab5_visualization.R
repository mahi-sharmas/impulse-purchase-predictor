# =============================================================================
# LAB 5: Data Visualization (Static)
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab2_cleaning.R (cleaned_data.rds)
# OUTPUT: 7 PNG plots in /content/outputs/plots/
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("ggplot2", "corrplot", "scales"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(ggplot2)
library(corrplot)
library(scales)

# ── Load data ─────────────────────────────────────────────────────────────────
df <- readRDS("/content/data/cleaned/cleaned_data.rds")

# Helper to save plots
save_plot <- function(filename, width=8, height=5) {
  ggsave(paste0("/content/outputs/plots/", filename),
         width=width, height=height, dpi=150)
  cat("✓ Saved:", filename, "\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Class Distribution — Impulse Buyer Count
# ─────────────────────────────────────────────────────────────────────────────
p1 <- ggplot(df, aes(x = impulse_buyer, fill = impulse_buyer)) +
  geom_bar(width = 0.5) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("No" = "#E07B54", "Yes" = "#4E9AF1")) +
  labs(title = "Impulse Buyer Class Distribution",
       x = "Impulse Buyer", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")
print(p1); save_plot("plot1_class_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Histogram — Total Purchase Amount by Impulse Class
# ─────────────────────────────────────────────────────────────────────────────
p2 <- ggplot(df, aes(x = total_purchase_amount, fill = impulse_buyer)) +
  geom_histogram(bins = 40, alpha = 0.8, position = "identity") +
  facet_wrap(~impulse_buyer, ncol = 2) +
  scale_fill_manual(values = c("No" = "#E07B54", "Yes" = "#4E9AF1")) +
  labs(title = "Total Purchase Amount Distribution by Impulse Buyer Class",
       x = "Total Purchase Amount", y = "Count", fill = "Impulse Buyer") +
  theme_minimal()
print(p2); save_plot("plot2_purchase_amount_histogram.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Box Plot — Product Price by Impulse Class
# ─────────────────────────────────────────────────────────────────────────────
p3 <- ggplot(df, aes(x = impulse_buyer, y = product_price, fill = impulse_buyer)) +
  geom_boxplot(outlier.alpha = 0.3, width = 0.5) +
  scale_fill_manual(values = c("No" = "#E07B54", "Yes" = "#4E9AF1")) +
  labs(title = "Product Price by Impulse Buyer Class",
       x = "Impulse Buyer", y = "Product Price") +
  theme_minimal() +
  theme(legend.position = "none")
print(p3); save_plot("plot3_product_price_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Bar Chart — Impulse Rate by Product Category
# ─────────────────────────────────────────────────────────────────────────────
p4 <- ggplot(df, aes(x = product_category, fill = impulse_buyer)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("No" = "#E07B54", "Yes" = "#4E9AF1")) +
  scale_y_continuous(labels = percent) +
  labs(title = "Impulse Buyer Rate by Product Category",
       x = "Product Category", y = "Proportion", fill = "Impulse Buyer") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p4); save_plot("plot4_category_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Bar Chart — Impulse Rate by Payment Method
# ─────────────────────────────────────────────────────────────────────────────
p5 <- ggplot(df, aes(x = payment_method, fill = impulse_buyer)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("No" = "#E07B54", "Yes" = "#4E9AF1")) +
  scale_y_continuous(labels = percent) +
  labs(title = "Impulse Buyer Rate by Payment Method",
       x = "Payment Method", y = "Proportion", fill = "Impulse Buyer") +
  theme_minimal()
print(p5); save_plot("plot5_payment_method_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: Correlation Heatmap (numeric features)
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols <- c("product_price", "quantity", "total_purchase_amount",
                  "customer_age", "returns")
cor_matrix <- cor(df[, numeric_cols], use = "complete.obs")

png("/content/outputs/plots/plot6_correlation_heatmap.png",
    width = 800, height = 700, res = 130)
corrplot(cor_matrix,
         method      = "color",
         type        = "upper",
         tl.col      = "black",
         tl.cex      = 0.9,
         addCoef.col = "black",
         number.cex  = 0.75,
         col         = colorRampPalette(c("#2166AC", "white", "#B2182B"))(200),
         title       = "Feature Correlation Heatmap",
         mar         = c(0, 0, 2, 0))
dev.off()
cat("✓ Saved: plot6_correlation_heatmap.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7: Box Plot — Quantity by Impulse Class
# ─────────────────────────────────────────────────────────────────────────────
p7 <- ggplot(df, aes(x = impulse_buyer, y = quantity, fill = impulse_buyer)) +
  geom_boxplot(width = 0.5) +
  scale_fill_manual(values = c("No" = "#E07B54", "Yes" = "#4E9AF1")) +
  labs(title = "Quantity Purchased by Impulse Buyer Class",
       x = "Impulse Buyer", y = "Quantity") +
  theme_minimal() +
  theme(legend.position = "none")
print(p7); save_plot("plot7_quantity_boxplot.png")

cat("\n=== LAB 5 COMPLETE ===\n")
cat("7 plots saved to /content/outputs/plots/\n")
cat("Next: Run lab6_dashboard.R\n")
