# ESE538 Time Series Forecasting Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive time series forecasting benchmark comparing classical statistical methods, gradient boosting models, and deep learning approaches for retail sales prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Feature Engineering](#feature-engineering)
- [Visualizations](#visualizations)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project implements a complete baseline for time series forecasting using the [Kaggle Store Sales dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). It compares **8 different models** across **3 forecast horizons** (7, 14, and 28 days) with rigorous evaluation methodology.

**Dataset**: Store sales data from Corporación Favorita, a large Ecuadorian grocery retailer
- **Training Period**: 2013-02-26 to 2016-12-31
- **Validation Period**: 2017-01-01 to 2017-04-30
- **Test Period**: 2017-05-01 to 2017-07-18
- **Total Observations**: 25,600 time series points
- **Stores**: 4 (sampled from 54 total)
- **Product Families**: 5 categories (GROCERY I, BEVERAGES, DAIRY, BREAD/BAKERY, MEAT)

## ✨ Key Features

- **🤖 8 Model Implementations**:
  - Classical: Seasonal Naive, ETS, SARIMAX
  - Gradient Boosting: LightGBM, XGBoost, CatBoost
  - Deep Learning: TCN, TCN++ (enhanced with GLU gates)

- **📊 Advanced Feature Engineering**:
  - 100+ features including lag features, rolling statistics, Fourier transforms, holiday effects
  - Feature ablation support for category-wise analysis

- **🎲 Probabilistic Forecasting**:
  - Quantile regression (q10, q50, q90)
  - Conformal prediction calibration for prediction intervals

- **📈 Rigorous Evaluation**:
  - Point metrics: RMSE, MAE, sMAPE, MASE
  - Probabilistic metrics: Coverage, CRPS, Pinball Loss
  - Statistical tests: Diebold-Mariano with HAC variance

- **🔧 Hyperparameter Optimization**:
  - Bayesian optimization using Optuna (20 trials, TPE sampler)

- **📉 Comprehensive Visualizations**:
  - Performance heatmaps, residual diagnostics, family-wise comparisons
  - 15+ publication-ready plots automatically generated

## 🏆 Results Summary

### Best Models by RMSE (Validation Set)

| Horizon | Best Model | RMSE | MASE | Improvement vs Naive |
|---------|-----------|------|------|---------------------|
| 7 days  | **TCN++** | 690.63 | 0.947 | **32.6%** ↑ |
| 14 days | **TCN++** | 751.66 | 0.993 | **12.8%** ↑ |
| 28 days | **LightGBM (Median)** | 769.31 | 0.952 | **14.0%** ↑ |

### Best Models by RMSE (Test Set)

| Horizon | Best Model | RMSE | MASE | Improvement vs Naive |
|---------|-----------|------|------|---------------------|
| 7 days  | **LightGBM (Median)** | 454.11 | 0.600 | **41.6%** ↑ |
| 14 days | **LightGBM (Median)** | 464.38 | 0.574 | **32.9%** ↑ |
| 28 days | **TCN** | 509.30 | 0.755 | **21.5%** ↑ |

### Key Findings

✅ **LightGBM** achieves best overall performance (MASE: 0.911, 8.9% better than Seasonal Naive baseline)
✅ **TCN++** demonstrates superior short-term forecasting with optimized architecture
✅ **Conformal Prediction** successfully calibrates prediction intervals: 71.9% → 80.1% coverage
✅ **Statistical Significance**: Tree-based models significantly outperform classical methods (Diebold-Mariano test, p < 0.05)
⚠️ **Classical Methods** (ETS, SARIMAX) struggle with complex non-linear patterns (MASE > 4.0)

## 📁 Project Structure

```
ESE_5380_proj/
│
├── 📓 ESE438_Project.ipynb    # Main notebook with full implementation
├── 📄 ESE4380_LQWZ.pdf         # Project report
│
├── 📊 results/                 # Auto-generated output directory
│   ├── point_metrics.csv       # Point forecast metrics
│   ├── interval_metrics.csv    # Probabilistic metrics
│   ├── dm_tests.csv            # Diebold-Mariano test results
│   ├── rmse_comparison.png     # Model comparison plots
│   ├── feature_importance.png  # Feature analysis
│   ├── family_*_heatmap.png    # Family-wise performance
│   ├── residual_analysis.png   # Diagnostic plots
│   └── ...                     # 15+ visualizations
│
├── 📋 README.md                # This file
├── 📦 requirements.txt         # Python dependencies
├── 📜 LICENSE                  # MIT License
├── 🚫 .gitignore              # Git ignore rules
└── 🤝 CONTRIBUTING.md         # Contribution guidelines
```

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU (optional, recommended for TCN models)
- **Kaggle API**: Account credentials for dataset download

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ESE_5380_proj.git
cd ESE_5380_proj
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Kaggle API**:
   - Download your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - Set permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac only
     ```

5. **Verify installation**:
```bash
python -c "import lightgbm, xgboost, catboost, torch; print('All packages installed successfully!')"
```

## 💻 Usage

### Running the Full Pipeline

**Option 1: Jupyter Notebook** (Recommended)
```bash
jupyter notebook ESE438_Project.ipynb
```

**Option 2: JupyterLab**
```bash
jupyter lab
```

**Option 3: Google Colab**
- Upload `ESE438_Project.ipynb` to Google Drive
- Open with Google Colab
- The notebook will auto-download the dataset

### Notebook Structure

The notebook is organized into **9 main sections**:

1. **Setup & Installation** - Package installation and imports
2. **Data Loading** - Automatic Kaggle dataset download
3. **Feature Engineering** - 100+ features with ablation support
4. **Hyperparameter Optimization** - Bayesian optimization (Optuna)
5. **Model Training** - All 8 models across 3 horizons
6. **Statistical Testing** - Diebold-Mariano significance tests
7. **Results Analysis** - Comprehensive metrics and rankings
8. **Feature Importance** - Top features and category analysis
9. **Visualization** - 15+ publication-ready plots

### Quick Start Example

```python
# Load data
from pathlib import Path
BASE_PATH = Path("data/raw")
tables = load_kaggle_tables(BASE_PATH)

# Feature engineering
panel_df, feature_df, feature_cols, target_cols, snaive_cols = prepare_feature_panel(
    tables=tables,
    target_stores=[1, 2, 3, 4],
    target_families=["GROCERY I", "BEVERAGES", "DAIRY", "BREAD/BAKERY", "MEAT"],
    horizons=[7, 14, 28],
    seasonal_period=7
)

# Train LightGBM with quantile regression
lgb_q10, lgb_q50, lgb_q90 = train_lightgbm_quantiles(
    X_train, y_train,
    X_val, y_val,
    params=LIGHTGBM_BASE_PARAMS,
    num_boost_round=1000,
    early_stopping_rounds=50
)

# Predict and evaluate
predictions = lgb_q50.predict(X_test)
rmse_score = rmse(y_test, predictions)
print(f"Test RMSE: {rmse_score:.2f}")
```

## 🤖 Models Implemented

### 1. Classical Statistical Methods

#### Seasonal Naive (P=7)
- **Type**: Baseline
- **Description**: Uses value from same day of week 7 days ago
- **Best for**: Establishing minimum performance threshold
- **Validation RMSE**: 1025.10 (H=7d)

#### ETS (Exponential Smoothing)
- **Type**: State-space model
- **Components**: Additive trend + seasonal (period=7)
- **Optimization**: Maximum likelihood estimation
- **Validation RMSE**: 8491.87 (H=7d) ⚠️ Poor performance

#### SARIMAX
- **Type**: Seasonal ARIMA with exogenous variables
- **Order**: (1,1,1) × (1,0,1,7)
- **Exogenous vars**: Oil price, holidays, promotions, temporal features
- **Validation RMSE**: 2898.61 (H=7d)

### 2. Gradient Boosting Models

#### LightGBM ⭐
- **Type**: Gradient boosting decision trees
- **Key Features**:
  - Quantile regression (α = 0.1, 0.5, 0.9)
  - Bayesian hyperparameter optimization (Optuna)
  - Conformal prediction calibration
- **Optimized Params**:
  - `learning_rate`: 0.087, `num_leaves`: 68, `max_depth`: 6
  - `feature_fraction`: 0.628, `bagging_fraction`: 0.643
- **Validation RMSE**: **714.73** (H=7d) 🏆
- **Test RMSE**: **454.11** (H=7d) 🏆

#### XGBoost
- **Type**: Extreme gradient boosting
- **Objective**: reg:squarederror
- **Params**: `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`
- **Validation RMSE**: 824.69 (H=7d)

#### CatBoost
- **Type**: Categorical boosting
- **Key Features**: Native categorical feature handling, ordered boosting
- **Params**: `iterations=1000`, `depth=6`, `learning_rate=0.05`
- **Validation RMSE**: 797.24 (H=7d)

### 3. Deep Learning Models

#### TCN (Temporal Convolutional Network)
- **Architecture**: 3-layer TCN with residual connections
- **Channels**: [64, 64, 32]
- **Key Features**:
  - Dilated causal convolutions (dilation: 1, 2, 4)
  - Embedding layers for store and family IDs
  - Input window: 56 days
- **Training**: Adam optimizer, MSE loss, early stopping
- **Validation RMSE**: 810.84 (H=7d)

#### TCN++ (Enhanced TCN) ⭐
- **Architecture**: 4-layer TCN with advanced components
- **Channels**: [64, 64, 64, 32]
- **Enhancements**:
  - **GLU (Gated Linear Units)**: Improved gradient flow
  - **BatchNorm**: Faster convergence and stability
  - **Data Standardization**: StandardScaler for features
  - **AdamW Optimizer**: Weight decay for regularization
  - **ReduceLROnPlateau Scheduler**: Adaptive learning rate
- **Validation RMSE**: **690.63** (H=7d) 🏆
- **Key Advantage**: Best short-term forecasting performance

## 📊 Evaluation Metrics

### Point Forecast Metrics

| Metric | Formula | Interpretation | Target |
|--------|---------|----------------|--------|
| **RMSE** | √(Σ(y - ŷ)² / n) | Penalizes large errors | Lower is better |
| **MAE** | Σ\|y - ŷ\| / n | Average absolute error | Lower is better |
| **sMAPE** | 100% × Σ\|y - ŷ\| / ((\|y\| + \|ŷ\|)/2) / n | Symmetric percentage error | Lower is better |
| **MASE** | MAE / MAE_naive | Scaled error vs baseline | < 1.0 beats baseline |

### Probabilistic Forecast Metrics

| Metric | Description | Formula | Target |
|--------|-------------|---------|--------|
| **Coverage** | Prediction interval accuracy | % observations within [q10, q90] | 80% |
| **CRPS** | Continuous Ranked Probability Score | Mean pinball loss across quantiles | Lower is better |
| **Pinball Loss** | Quantile regression loss | max(α(y-ŷ), (α-1)(y-ŷ)) | Lower is better |

### Statistical Significance Tests

#### Diebold-Mariano Test
- **Purpose**: Compare forecast accuracy of two models
- **Null Hypothesis**: Models have equal predictive accuracy
- **Test Statistic**: DM = d̄ / √(Var(d̄))
- **Variance**: Newey-West HAC estimator (accounts for autocorrelation)
- **Interpretation**:
  - DM > 0 & p < 0.05: Model 2 is significantly better
  - DM < 0 & p < 0.05: Model 1 is significantly better
  - p ≥ 0.05: No significant difference

**Key Result**: LightGBM significantly outperforms Seasonal Naive (DM = -3.487, p = 0.0005) at H=7d

## 🔧 Feature Engineering

The pipeline generates **98 features** across **8 categories**:

### 1. Lag Features (Total Importance: 22,792)
```python
lag_list = [1, 7, 14, 28, 56]
# For each lag: sales, onpromotion, store_transactions
# Top contributor: lag_sales_14 (importance: 9,398)
```

### 2. Rolling Statistics (Total Importance: 4,706)
```python
windows = [7, 14, 28, 56]
# For each window: mean, std, median, max, min, CV
# Example: roll_mean_sales_7, roll_std_sales_28
```

### 3. Differential Features (Total Importance: 522)
```python
diff_lags = [1, 7, 14]
# First differences: diff_sales_7, diff_onpromo_14
```

### 4. Temporal Features (Total Importance: 2,158)
```python
# Cyclical: dayofweek, month, weekofyear
# Binary: is_weekend, is_month_end
# Continuous: month_progress (dayofmonth / days_in_month)
```

### 5. Fourier Features (Total Importance: 3,410)
```python
# Weekly: sin(2π × dayofweek / 7), cos(...)
# Bi-weekly: sin(2π × dayofweek / 14), cos(...)
# Quarterly: sin(2π × dayofyear / 91.25), cos(...)
# Yearly: sin(2π × dayofyear / 365.25), cos(...)
```

### 6. Holiday Features (Total Importance: 1,173)
```python
# Binary indicators: is_national_holiday, is_regional_holiday, is_local_holiday
# Distance metrics: days_since_any_holiday, days_to_any_holiday
# Windows: is_holiday_next_7d, is_post_holiday_7d
```

### 7. Store-Level Features (Total Importance: 932)
```python
# Aggregations: store_daily_sales, store_sales_rank_pct
# Market share: store_sales_share
# Rolling: store_rolling_sales_7, store_rolling_sales_28
```

### 8. Family-Level Features (Total Importance: 1,042)
```python
# Aggregations: family_daily_sales, family_sales_share
# Rolling: family_roll_mean_7, family_roll_mean_28
```

### Top 10 Most Important Features (LightGBM)

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | sales | 23,959 | Target |
| 2 | lag_sales_14 | 9,398 | Lag |
| 3 | lag_sales_7 | 6,852 | Lag |
| 4 | log_sales_plus1 | 5,888 | Transform |
| 5 | lag_sales_28 | 2,742 | Lag |
| 6 | lag_sales_56 | 2,198 | Lag |
| 7 | roll_median_sales_7 | 1,239 | Rolling |
| 8 | fourier_quarter_cos | 928 | Fourier |
| 9 | roll_median_sales_56 | 841 | Rolling |
| 10 | month_progress | 777 | Temporal |

## 📈 Visualizations

All visualizations are automatically generated in `results/` directory:

### Performance Comparison
- `rmse_comparison.png` - RMSE by model and horizon (validation & test)
- `mase_comparison.png` - MASE comparison with baseline threshold
- `model_comparison_log.png` - Log-scale performance overview
- `model_comparison_zoomed.png` - Zoomed view of top performers

### Detailed Analysis
- `performance_heatmap.png` - RMSE heatmap (model × horizon)
- `family_rmse_heatmap.png` - Performance by product family
- `family_mase_heatmap.png` - MASE by family (color-coded: green < 1.0 < red)

### Feature Analysis
- `feature_importance.png` - Top 20 features bar chart
- `feature_categories.png` - Category-wise importance pie chart

### Model Diagnostics
- `residual_analysis.png` - 4-panel diagnostic plot:
  - Residual histogram
  - Q-Q plot (normality test)
  - Residuals vs fitted values
  - Residuals over time
- `predictions_store1_GROCERY I.png` - Sample forecast visualization

### Probabilistic Forecasting
- `coverage_comparison.png` - Prediction interval coverage (target: 80%)

## 📚 Citation

If you use this code in your research or projects, please cite:

```bibtex
@misc{ese538_forecasting_2024,
  title={ESE538 Complete Time Series Forecasting Baseline},
  author={LQWZ Team},
  year={2024},
  institution={University of Pennsylvania},
  course={ESE538: Time Series Analysis},
  url={https://github.com/yourusername/ESE_5380_proj}
}
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **Dataset**: [Kaggle Competition License](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules)
- **LightGBM**: MIT License
- **XGBoost**: Apache License 2.0
- **CatBoost**: Apache License 2.0
- **PyTorch**: BSD License

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) by Corporación Favorita
- **Course**: ESE538 Time Series Analysis, University of Pennsylvania
- **Instructor**: [Course Instructor Name]
- **Libraries**:
  - [LightGBM](https://lightgbm.readthedocs.io/) - Microsoft
  - [XGBoost](https://xgboost.readthedocs.io/) - DMLC
  - [CatBoost](https://catboost.ai/) - Yandex
  - [statsmodels](https://www.statsmodels.org/) - statsmodels developers
  - [PyTorch](https://pytorch.org/) - Meta AI
  - [Optuna](https://optuna.org/) - Preferred Networks

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Improvement
- [ ] Implement additional models (Transformer, N-BEATS, Prophet)
- [ ] Add ensemble methods (stacking, blending)
- [ ] Extend to multivariate forecasting
- [ ] Implement online learning capabilities
- [ ] Add real-time prediction API

## 📞 Contact

For questions, issues, or feedback:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/ESE_5380_proj/issues)
- **Email**: lcb0105@seas.upenn.edu
- **Course**: ESE538, University of Pennsylvania

---

**Last Updated**: December 2025
**Status**: ✅ Production Ready
**Notebook Runtime**: ~30 minutes (with GPU)
**Dataset Size**: ~21.4 MB (compressed)
