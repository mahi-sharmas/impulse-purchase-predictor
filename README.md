# Impulse Purchase Prediction in E-commerce
> End-to-end machine learning project built in R — binary classification on 250,000 e-commerce transactions

---

## Results

| Model | Accuracy | AUC | F1 Score |
|---|---|---|---|
| Logistic Regression | 83.56% | 0.9150 | 0.7876 |
| Random Forest | 88.34% | 0.9589 | 0.8590 |
| **Decision Tree** ✓ | **89.21%** | **0.9633** | **0.8722** |

**Best model:** Decision Tree — highest accuracy, AUC, and F1 across all three classifiers.

---

## Project Overview

Retailers lose significant revenue when they fail to identify customers likely to make unplanned, high-value purchases. This project builds a binary classifier to predict whether an e-commerce customer is an **Impulse Buyer** based on their purchasing behavior.

- **Dataset:** 250,000 e-commerce transactions (13 raw features)
- **Target:** `impulse_buyer` — engineered binary label (top 40% spenders = Yes)
- **Language:** R (Google Colab)
- **Structure:** 9 modular lab scripts covering the full ML pipeline

---

## Project Structure

```
impulse-purchase-predictor/
│
├── scripts/
│   ├── lab1_import.R          # Data loading and inspection
│   ├── lab2_cleaning.R        # Missing values, type conversion, deduplication
│   ├── lab3_features.R        # Feature engineering and scaling
│   ├── lab4_eda.R             # Statistical summaries and correlation analysis
│   ├── lab5_visualization.R   # Static plots (ggplot2)
│   ├── lab6_dashboard.R       # Interactive Plotly dashboard
│   ├── lab7_regression.R      # Linear + Logistic Regression
│   ├── lab8_classification.R  # Decision Tree + Random Forest + comparison
│   └── lab9_clustering.R      # K-Means + Hierarchical clustering
│
├── data/
│   ├── raw/                   # Original CSV files
│   ├── cleaned/               # Cleaned dataset (cleaned_data.rds)
│   └── engineered/            # Train/test splits (scaled)
│
├── outputs/
│   ├── plots/                 # All saved charts (PNG)
│   └── models/                # Saved model objects (.rds)
│
└── dashboard/
    └── dashboard.html         # Standalone interactive dashboard
```

---

## Pipeline

### Lab 1 — Data Import
- Loaded 250,000 rows × 13 columns from CSV (inside ZIP)
- Inspected data types, dimensions, and summary statistics

### Lab 2 — Data Cleaning
- Standardized column names to `snake_case` using `janitor`
- Engineered binary target: `impulse_buyer = 1` if `total_purchase_amount > 60th percentile`
- Imputed 47,596 missing values in `returns` column with 0
- Dropped irrelevant columns: `customer_id`, `customer_name`, `age` (duplicate), `churn`
- Converted categorical columns to factors

### Lab 3 — Feature Engineering
- Extracted date features from `purchase_date`: `hour_of_day`, `day_of_week`, `is_weekend`, `month`
- Created behavioral features: `price_per_item`, `is_high_quantity`
- One-hot encoded categorical variables
- Applied Z-score scaling (fit on train, applied to test — no data leakage)
- 70/30 train-test split with stratification

### Lab 4 — EDA
- Class distribution, correlation matrix, group-level summaries
- Identified `total_purchase_amount` and `product_price` as strongest predictors

### Lab 5 — Static Visualization
- Spending distribution histogram
- Impulse rate by product category (bar chart)
- Price distribution by impulse class (box plot)
- Feature correlation heatmap

### Lab 6 — Interactive Dashboard
- Built with Plotly — saved as standalone `dashboard.html`
- Includes: impulse rate by category, spending distribution, age vs spend scatter, payment method breakdown

### Lab 7 — Regression Models
- **Linear Regression** on `total_purchase_amount` → RMSE: 0.677, R²: 0.539
- **Logistic Regression** with 5-fold CV → Accuracy: 83.56%, AUC: 0.915

### Lab 8 — Classification Models
- **Decision Tree** (tuned `cp`) → Accuracy: 89.21%, AUC: 0.9633 ✓ Best
- **Random Forest** (tuned `mtry`) → Accuracy: 88.34%, AUC: 0.9589
- ROC curve comparison and feature importance plots generated

### Lab 9 — Clustering
- K-Means with elbow + silhouette method to select optimal k
- Hierarchical clustering (Ward's linkage) on 1,000-row sample
- Cluster profiles: Impulse Spenders, Bulk Buyers, Premium Shoppers, Casual Browsers

---

## Key Techniques

- Target engineering from raw transactional data
- Train/test split with Z-score scaling (no leakage)
- 5-fold cross-validation for logistic regression
- Hyperparameter tuning (`cp` for Decision Tree, `mtry` for Random Forest)
- Silhouette scoring for optimal cluster selection
- ROC-AUC as primary evaluation metric (handles class imbalance better than accuracy)

---

## How to Run

Each lab is a standalone R script. Run them in order in **Google Colab (R kernel)**:

```
lab1 → lab2 → lab3 → lab4 → lab5 → lab6 → lab7 → lab8 → lab9
```

Each script installs its own dependencies and saves outputs before the next script begins.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | R |
| Data Wrangling | `dplyr`, `tidyr`, `janitor`, `lubridate` |
| Visualization | `ggplot2`, `plotly` |
| Machine Learning | `caret`, `rpart`, `randomForest` |
| Evaluation | `pROC`, `caret` |
| Clustering | `cluster`, `factoextra` |
| Environment | Google Colab (R kernel) |

---

## Dataset

- **Source:** E-commerce shopper behavior dataset (CSV inside ZIP)
- **Size:** 250,000 rows × 13 columns
- **Key columns:** `product_price`, `quantity`, `total_purchase_amount`, `payment_method`, `product_category`, `customer_age`, `gender`, `returns`, `purchase_date`
