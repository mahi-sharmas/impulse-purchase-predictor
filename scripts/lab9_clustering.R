# =============================================================================
# LAB 9: Clustering Techniques
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab2_cleaning.R (cleaned_data.rds)
# NOTE: Clustering is unsupervised — target column is REMOVED before fitting
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("cluster", "factoextra", "dplyr", "ggplot2"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(cluster)
library(factoextra)
library(dplyr)
library(ggplot2)

# ── Load cleaned data ─────────────────────────────────────────────────────────
df <- readRDS("/content/data/cleaned/cleaned_data.rds")
cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n")

target_col <- "impulse_buyer"

# Behavioral numeric features for clustering
# (purchase_date was already dropped in Lab 3, but here we load cleaned data
#  which still has purchase_date — extract hour here if needed)
cluster_features <- c("product_price", "quantity",
                       "total_purchase_amount", "customer_age", "returns")

cat("Clustering features:", paste(cluster_features, collapse=", "), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Prepare Clustering Data
# ─────────────────────────────────────────────────────────────────────────────
target_vec <- df[[target_col]]          # keep for interpretation later
df_cluster <- df[, cluster_features]   # only numeric behavioral features

# Z-score scale — critical for distance-based algorithms
df_scaled <- as.data.frame(scale(df_cluster))
cat("✓ Features scaled (Z-score)\n")

# =============================================================================
# STEP 2: K-MEANS CLUSTERING
# =============================================================================
cat("\n========== K-MEANS CLUSTERING ==========\n")
set.seed(42)

# Use a fixed 5k sample for all evaluation (elbow + silhouette)
# Keeps km$cluster and dist() on the SAME rows — fixes size mismatch error
eval_idx <- sample(nrow(df_scaled), 5000)
df_eval  <- df_scaled[eval_idx, ]

# Elbow plot (on sample — fast and representative)
wcss <- sapply(1:10, function(k) {
  kmeans(df_eval, centers=k, nstart=10, iter.max=100)$tot.withinss
})

p_elbow <- ggplot(data.frame(k=1:10, wcss=wcss), aes(x=k, y=wcss)) +
  geom_line(color="#4E9AF1", linewidth=1) +
  geom_point(color="#4E9AF1", size=3) +
  scale_x_continuous(breaks=1:10) +
  labs(title="Elbow Method — Optimal Number of Clusters",
       x="Number of Clusters (k)", y="Within-Cluster Sum of Squares") +
  theme_minimal()
print(p_elbow)
ggsave("/content/outputs/plots/elbow_plot.png",
       plot=p_elbow, width=7, height=4, dpi=150)
cat("✓ Elbow plot saved\n")

# Silhouette scores (k = 2 to 8) — same 5k sample used for both km and dist
dist_eval  <- dist(df_eval)
sil_scores <- sapply(2:8, function(k) {
  km <- kmeans(df_eval, centers=k, nstart=10, iter.max=100)
  mean(silhouette(km$cluster, dist_eval)[, 3])
})
optimal_k <- which.max(sil_scores) + 1
cat("Silhouette scores for k=2..8:", round(sil_scores, 3), "\n")
cat("Optimal k by silhouette     :", optimal_k, "\n")

# Fit final K-Means on full data (lower nstart avoids Quick-TRANSfer warnings)
set.seed(42)
km_model     <- kmeans(df_scaled, centers=optimal_k, nstart=10, iter.max=100)
df$km_cluster <- as.factor(km_model$cluster)
cat("✓ K-Means fitted (k=", optimal_k, ")\n")
print(table(df$km_cluster))

# =============================================================================
# STEP 3: HIERARCHICAL CLUSTERING
# =============================================================================
cat("\n========== HIERARCHICAL CLUSTERING ==========\n")

# Use a 1000-row sample for dendrogram (250k rows → unreadable + OOM risk)
set.seed(42)
n_sample   <- min(1000, nrow(df_scaled))
sample_idx <- sample(nrow(df_scaled), n_sample)
df_sample  <- df_scaled[sample_idx, ]

dist_matrix <- dist(df_sample, method="euclidean")
hc_model    <- hclust(dist_matrix, method="ward.D2")

png("/content/outputs/plots/dendrogram.png", width=1000, height=600, res=120)
fviz_dend(hc_model,
          k        = optimal_k,
          k_colors = c("#4E9AF1","#E07B54","#2CA02C","#FF7F0E","#9467BD")[1:optimal_k],
          rect      = TRUE,
          rect_fill = TRUE,
          main      = paste0("Hierarchical Clustering (Ward's, k=", optimal_k, ") — Sample n=", n_sample),
          ggtheme   = theme_minimal())
dev.off()
cat("✓ Dendrogram saved\n")

# Cut tree
hc_labels <- cutree(hc_model, k=optimal_k)
cat("Hierarchical cluster sizes (", n_sample, "row sample):\n")
print(table(hc_labels))

# =============================================================================
# STEP 4: CLUSTER INTERPRETATION
# =============================================================================
cat("\n========== CLUSTER PROFILES ==========\n")

cluster_profile <- df %>%
  group_by(km_cluster) %>%
  summarise(
    n                     = n(),
    avg_product_price     = round(mean(product_price,          na.rm=T), 2),
    avg_quantity          = round(mean(quantity,               na.rm=T), 2),
    avg_total_purchase    = round(mean(total_purchase_amount,  na.rm=T), 2),
    avg_age               = round(mean(customer_age,           na.rm=T), 1),
    return_rate_pct       = round(mean(returns,                na.rm=T) * 100, 1),
    .groups = "drop"
  )
cat("--- Cluster Profile Table ---\n")
print(as.data.frame(cluster_profile))

cat("\n--- Impulse Buyer % per Cluster ---\n")
cross_tab <- df %>%
  group_by(km_cluster) %>%
  summarise(impulse_rate = round(mean(impulse_buyer == "Yes") * 100, 1),
            .groups="drop")
print(as.data.frame(cross_tab))

# =============================================================================
# STEP 5: VISUALIZE
# =============================================================================
cat("\n========== CLUSTER VISUALIZATIONS ==========\n")

# PCA scatter plot
png("/content/outputs/plots/cluster_pca.png", width=900, height=700, res=120)
fviz_cluster(km_model,
             data      = df_scaled,
             geom      = "point",
             ellipse   = TRUE,
             ellipse.type = "convex",
             pointsize = 0.8,
             ggtheme   = theme_minimal(),
             main      = paste0("K-Means Clusters (k=", optimal_k, ") — PCA"))
dev.off()
cat("✓ PCA cluster plot saved\n")

# Impulse rate per cluster bar chart
p_cr <- ggplot(cross_tab, aes(x = km_cluster, y = impulse_rate, fill = km_cluster)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(impulse_rate, "%")), vjust = -0.4, size = 4) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Impulse Buyer Rate by Cluster",
       x = "Cluster", y = "Impulse Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none")

print(p_cr)
ggsave("/content/outputs/plots/cluster_impulse_rate.png",
       plot = p_cr, width = 7, height = 5, dpi = 150)
cat("✓ Cluster impulse rate chart saved\n")

cat("\n--- Suggested Cluster Names (based on profile table above) ---\n")
cat("  High avg_total_purchase + high impulse_rate → 'Impulse Spenders'\n")
cat("  High avg_quantity + low product_price       → 'Bulk Bargain Buyers'\n")
cat("  High product_price + low quantity           → 'Premium Shoppers'\n")
cat("  Low spend + low returns                     → 'Casual Browsers'\n")

cat("\n=== LAB 9 COMPLETE ===\n")
cat("All outputs in /content/outputs/\n")
cat("\n✓ ALL 9 LABS COMPLETE — Project done!\n")
