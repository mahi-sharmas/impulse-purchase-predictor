# =============================================================================
# LAB 6: Interactive Visualization & Dashboard (Plotly HTML)
# Project: Impulse Purchase Prediction in E-commerce
# Runtime: Google Colab (R kernel)
# =============================================================================
# DEPENDS ON: lab2_cleaning.R (cleaned_data.rds)
# OUTPUT: /content/dashboard/dashboard.html (open in any browser)
# =============================================================================

# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c("plotly", "htmlwidgets", "htmltools", "dplyr"), quiet = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(plotly)
library(htmlwidgets)
library(htmltools)
library(dplyr)

# ── Load data ─────────────────────────────────────────────────────────────────
df <- readRDS("/content/data/cleaned/cleaned_data.rds")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Impulse Buyer Class Distribution
# ─────────────────────────────────────────────────────────────────────────────
class_df <- as.data.frame(table(df$impulse_buyer))
names(class_df) <- c("class", "count")

chart1 <- plot_ly(class_df,
                  x = ~class, y = ~count,
                  type   = "bar",
                  color  = ~class,
                  colors = c("#E07B54", "#4E9AF1"),
                  text   = ~count, textposition = "outside") %>%
  layout(title      = "Impulse Buyer Class Distribution",
         xaxis      = list(title = "Impulse Buyer"),
         yaxis      = list(title = "Count"),
         showlegend = FALSE)

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Impulse Rate by Product Category
# ─────────────────────────────────────────────────────────────────────────────
cat_rate <- df %>%
  group_by(product_category) %>%
  summarise(impulse_rate = round(mean(impulse_buyer == "Yes") * 100, 1),
            .groups = "drop") %>%
  arrange(desc(impulse_rate))

chart2 <- plot_ly(cat_rate,
                  x = ~impulse_rate,
                  y = ~reorder(product_category, impulse_rate),
                  type        = "bar",
                  orientation = "h",
                  marker      = list(color = "#4E9AF1"),
                  text        = ~paste0(impulse_rate, "%"),
                  textposition = "outside") %>%
  layout(title = "Impulse Buyer Rate by Product Category",
         xaxis = list(title = "Impulse Rate (%)"),
         yaxis = list(title = ""))

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Total Purchase Amount vs. Product Price (scatter)
# Colored by impulse buyer class — shows spending profile
# ─────────────────────────────────────────────────────────────────────────────

# Sample 3000 rows for readability (full 250k would crowd the scatter)
set.seed(42)
df_sample <- df[sample(nrow(df), min(3000, nrow(df))), ]

chart3 <- plot_ly(df_sample,
                  x     = ~product_price,
                  y     = ~total_purchase_amount,
                  color = ~impulse_buyer,
                  colors = c("#E07B54", "#4E9AF1"),
                  type  = "scatter",
                  mode  = "markers",
                  marker = list(size = 5, opacity = 0.5),
                  hovertemplate = paste0(
                    "<b>Price:</b> %{x}<br>",
                    "<b>Total:</b> %{y}<br>",
                    "<extra></extra>"
                  )) %>%
  layout(title  = "Product Price vs. Total Purchase Amount",
         xaxis  = list(title = "Product Price"),
         yaxis  = list(title = "Total Purchase Amount"),
         legend = list(title = list(text = "Impulse Buyer")))

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: Impulse Rate by Payment Method
# ─────────────────────────────────────────────────────────────────────────────
pay_rate <- df %>%
  group_by(payment_method) %>%
  summarise(impulse_rate = round(mean(impulse_buyer == "Yes") * 100, 1),
            .groups = "drop")

chart4 <- plot_ly(pay_rate,
                  x = ~payment_method,
                  y = ~impulse_rate,
                  type   = "bar",
                  marker = list(color = "#E07B54"),
                  text   = ~paste0(impulse_rate, "%"),
                  textposition = "outside") %>%
  layout(title = "Impulse Buyer Rate by Payment Method",
         xaxis = list(title = "Payment Method"),
         yaxis = list(title = "Impulse Rate (%)"))

# ─────────────────────────────────────────────────────────────────────────────
# Assemble Dashboard
# ─────────────────────────────────────────────────────────────────────────────
dashboard <- tagList(
  tags$h1("Impulse Purchase Prediction — Dashboard",
          style = "font-family: Arial, sans-serif; text-align: center;
                   padding: 24px 0 4px; color: #2c3e50;"),
  tags$p("E-commerce Shopper Behavior | 250,000 Transactions",
         style = "font-family: Arial; text-align: center; color: #7f8c8d;
                  margin-bottom: 20px;"),
  tags$div(
    style = "display: flex; flex-wrap: wrap; justify-content: center; gap: 12px;
             padding: 0 12px;",
    tags$div(style = "width: 48%; min-width: 400px;", as.widget(chart1)),
    tags$div(style = "width: 48%; min-width: 400px;", as.widget(chart2)),
    tags$div(style = "width: 48%; min-width: 400px;", as.widget(chart3)),
    tags$div(style = "width: 48%; min-width: 400px;", as.widget(chart4))
  )
)

# ─────────────────────────────────────────────────────────────────────────────
# Save as self-contained HTML
# ─────────────────────────────────────────────────────────────────────────────
dir.create("/content/dashboard", showWarnings = FALSE)
save_html(dashboard, "/content/dashboard/dashboard.html")

cat("✓ Dashboard saved to /content/dashboard/dashboard.html\n")
cat("  → Download and open in your browser\n")

cat("\n=== LAB 6 COMPLETE ===\n")
cat("Next: Run lab7_regression.R\n")
