# Retention ROI: Churn Prediction and Customer Lifetime Value for E-Commerce

**Springboard Data Science Career Track — Capstone 3**

> Predicting which customers will churn — and estimating what retaining them is worth — in a non-contractual e-commerce setting.

---

## Project Overview

In subscription businesses, churn is explicit: a customer cancels and you know immediately. In e-commerce, it's invisible. Customers simply stop buying, and you only find out weeks later when they don't come back.

This project addresses that challenge using the UCI Online Retail II dataset (~700K transactions, ~5,800 unique customers, Dec 2009–Dec 2011). The goal: build a churn prediction model precise enough to drive retention spend decisions, and lay the groundwork for a full Customer Lifetime Value (CLV) estimation system.

**Primary business question:** Which customers are likely to churn in the next 90 days, and how should that probability inform retention investment?

---

## Results

| Model | PR-AUC | ROC-AUC | Brier Score |
|---|---|---|---|
| **Logistic Regression** | **0.7151** | **0.8032** | 0.1742 |
| XGBoost | 0.7056 | 0.7907 | 0.1803 |
| Random Forest | 0.7044 | 0.7980 | **0.1690** |

Logistic Regression achieved the best discrimination (PR-AUC, ROC-AUC), likely because log1p preprocessing linearized the dominant RFM relationships and the training set (~4,600 customers) is small enough that regularized linear models outperform ensembles. Random Forest had the best calibration (Brier score), which matters for CLV dollar estimates downstream.

**Primary metric: PR-AUC** (Precision-Recall AUC), chosen over accuracy because the churn positive rate (~33%) creates a misleading accuracy baseline and because precision-recall tradeoffs directly govern retention ROI.

---

## Approach

### Problem Framing

This is a **non-contractual, discrete-time survival** problem. A "churned" customer is operationally defined as one who makes no purchase in the 90 days following a calibration window — not because they cancelled, but because they went silent.

**Target variable:** `repurchase_90d` — binary label derived from holdout period (Feb–Aug 2011), with all features derived from the calibration period (Dec 2009–Jan 2011). This time-based split prevents any label leakage.

### Feature Engineering (20 Features)

Features are grouped into three conceptual categories:

**RFM Core:**
- `recency_days` — days since last purchase (end of calibration)
- `frequency` — total number of orders
- `monetary_total`, `monetary_avg_basket` — revenue metrics

**Behavioral:**
- `n_unique_products`, `n_unique_categories_proxy` — catalog breadth
- `basket_size_avg`, `basket_size_std` — basket consistency
- `return_rate`, `n_cancel_invoices` — return behavior
- `avg_unit_price` — price sensitivity
- `weekend_share` — shopping timing pattern

**Temporal/Loyalty:**
- `tenure_days` — customer lifespan
- `interpurchase_time_mean`, `interpurchase_time_max` — purchase rhythm
- `last_gap_days`, `last_vs_avg_gap_ratio` — recency relative to personal baseline
- `is_uk` — geographic segment
- `price_tier_mid`, `price_tier_premium` — customer segment dummies

### Preprocessing

- **Imputation:** `interpurchase_time_*` → `tenure_days`; `last_vs_avg_gap_ratio` → 1.0 (neutral); `basket_size_std` → 0.0
- **Scaling:** `log1p` transformation on 13 right-skewed features, then `StandardScaler`; `StandardScaler` only on 4 bounded features; binary features passed through unchanged
- **Split:** Stratified 80/20 train/test split at customer level (`random_state=42`)

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_acquisition.ipynb` | Load UCI dataset, initial inspection and quality assessment |
| `02_eda.ipynb` | Univariate distributions, bivariate analysis, correlation and mutual information |
| `03_preprocessing.ipynb` | Feature engineering, imputation, log1p scaling, stratified train/test split |
| `04_modeling.ipynb` | Logistic Regression, Random Forest, and XGBoost training, evaluation, and comparison |

---

## Repository Structure

```
.
├── data/
│   ├── raw/                        # UCI Online Retail II (.xlsx)
│   └── processed/                  # Train/test splits output by 03_preprocessing.ipynb
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── X_train_lr.csv          # Scaled version for Logistic Regression
│       ├── X_test_lr.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   └── preprocessor_lr.pkl         # Fitted sklearn pipeline (log1p + StandardScaler)
├── 01_data_acquisition.ipynb
├── 02_eda.ipynb
├── 03_preprocessing.ipynb
├── 04_modeling.ipynb
├── model_metrics.csv               # Full model comparison table + hyperparameters
├── Capstone_Final_Report.pdf       # Written report with embedded figures
├── Capstone_Final_Presentation.pptx
└── README.md
```

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter reportlab python-pptx
```

Tested with Python 3.10+.

### Running the Notebooks

Run notebooks in order (01 → 04). Each notebook reads outputs from the previous stage.

```bash
jupyter notebook
```

> **Note:** `03_preprocessing.ipynb` writes to `data/processed/` and `models/`. Those directories are expected to exist before running — the notebook's `os.makedirs(..., exist_ok=True)` calls will create them automatically.

---

## Dataset

**UCI Online Retail II**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- Transactions: ~1M rows (Dec 2009–Dec 2011)
- After cleaning: ~700K valid transactions, ~5,800 unique customers
- Geography: Primarily UK-based, with international customers

The dataset contains invoice-level records: invoice number, stock code, description, quantity, invoice date, unit price, and customer ID.

---

## Key Design Decisions

**Why PR-AUC over accuracy?**
With ~33% churn rate, a model predicting "no churn" for all customers gets 67% accuracy while catching zero churners. PR-AUC measures performance where it matters: identifying the minority class precisely enough to act on it.

**Why a time-based calibration/holdout split?**
E-commerce data has natural temporal structure. Features are derived from historical behavior (Dec 2009–Jan 2011); labels are derived from a future holdout window (Feb–Aug 2011). This prevents any label information from leaking into features, which would inflate model performance estimates.

**Why did Logistic Regression outperform XGBoost?**
Two factors: (1) ~4,600 training customers is a small dataset where regularized linear models often match or beat ensembles, and (2) log1p preprocessing linearized the RFM relationships that dominate churn prediction, reducing the nonlinear advantage of gradient boosting.

**Calibration matters for CLV**
Predicted probabilities feed directly into expected CLV calculations (probability × expected revenue). Poor calibration means bad dollar estimates even if discrimination is good. Random Forest had the best Brier score (0.1690) — isotonic regression or Platt scaling would further improve probability calibration for production use.

---

## Business Application

The model outputs a churn probability score for each customer. Combined with estimated CLV, this enables a 2×2 retention strategy matrix:

| | **High CLV** | **Low CLV** |
|---|---|---|
| **High Churn Risk** | Priority intervention — personalized outreach, loyalty rewards | Light-touch win-back — low-cost email campaign |
| **Low Churn Risk** | Nurture — upsell, cross-sell | Harvest — no spend, monitor |

A 40% precision threshold (configurable) ensures retention spend is only triggered when the model is reasonably confident, maintaining positive expected ROI on intervention costs.

---

## Author

Justin Ali · [justin.ali.data@gmail.com](mailto:justin.ali.data@gmail.com)  
Springboard Data Science Career Track — Capstone 3
