# Demand Forecasting with XGBoost and SHAP Explainability

## Project Overview

This project demonstrates end-to-end machine learning capabilities for time-series forecasting. Using a sample retail demand dataset, I built and evaluated multiple forecasting models, implemented explainability analysis, and created professional documentation.

**Purpose**: Technical skills demonstration for portfolio

**Key Skills Demonstrated**:
- Time-series feature engineering
- Machine learning model development and comparison
- Model evaluation with proper cross-validation
- Explainable AI using SHAP
- Professional code documentation and visualization

---

## Dataset

**Sample retail demand dataset** with the following characteristics:
- **Time Period**: 2 years (2022-2023)
- **Granularity**: Daily observations
- **Records**: 36,500 data points
- **Products**: 50 items across multiple categories

**Features**:
- Historical demand (target variable)
- Pricing information (product price, competitor price)
- Promotional indicators (campaigns, holidays)
- Temporal features (day of week, month)
- External factors (temperature, stock levels)

*Note: This is demonstration data used to showcase technical implementation.*

---

## Technical Approach

### 1. Feature Engineering

Created time-series specific features to capture demand patterns:

**Lag Features**:
- `demand_lag_1`, `demand_lag_7`, `demand_lag_14`, `demand_lag_30`
- Captures recent sales history

**Rolling Averages**:
- 7-day and 30-day rolling means
- Smooths out noise and identifies trends

**Temporal Encoding**:
- Day of week, month, holiday indicators
- Captures seasonality and calendar effects

### 2. Model Development

Implemented and compared three approaches:

| Model | Cross-Validation MAE | Test MAE | Notes |
|-------|---------------------|----------|-------|
| Linear Regression | 65.4 | 66.2 | Baseline model |
| Random Forest | 40.2 | 41.8 | Ensemble method |
| **XGBoost** | **36.6** | **38.6** | **Best performance** |

**Why XGBoost performed best**:
- Handles non-linear relationships effectively
- Robust to outliers
- Captures complex feature interactions
- Provides feature importance

### 3. Model Validation

**Time-Series Cross-Validation**:
- 5-fold time-series split (respects temporal ordering)
- Prevents data leakage from future to past
- Validates consistency across different time periods

**Test Set Evaluation**:
- Chronological 70/30 train/test split
- Final performance verified on unseen 2023 data
- MAE: 38.6 units (close to CV score of 36.6, indicating minimal overfitting)

**Metrics Used**:
- Mean Absolute Error (MAE) - primary metric
- Root Mean Squared Error (RMSE)
- R² Score

### 4. Explainability with SHAP

Implemented SHAP (SHapley Additive exPlanations) to interpret model predictions:

**Global Feature Importance** - Which features matter most overall?
- Identifies key drivers across all predictions
- Quantifies relative feature contribution

**Individual Predictions** - Why did the model make this specific prediction?
- Waterfall plots show feature-by-feature attribution
- Enables validation and debugging of predictions

---

## Key Technical Findings

### SHAP Feature Importance Analysis

The model's SHAP analysis reveals the following feature importance hierarchy:

| Rank | Feature | Mean \|SHAP Value\| | Interpretation |
|------|---------|---------------------|----------------|
| 1 | demand_rolling_avg_7 | 74.5 | 7-day smoothed trend is strongest predictor |
| 2 | has_promotion | 27.8 | Promotional activity has significant impact |
| 3 | day_of_week_6 (Sat) | 21.1 | Weekend patterns are important |
| 4 | day_of_week_5 (Fri) | 20.9 | Friday also shows elevated demand |
| 5 | demand_rolling_avg_30 | 17.5 | Longer-term trends add predictive value |
| 6 | demand_lag_7 | 10.6 | Weekly lag captures additional signal |
| 7 | is_holiday | 9.8 | Holiday effects are measurable |
| 8 | demand_lag_1 | 6.0 | Recent daily sales provide minor signal |

### Feature Category Breakdown

**Rolling Averages (~50% of total SHAP importance)**:
- Smoothed historical trends dominate predictions
- 7-day rolling average is the single strongest feature
- 30-day average captures longer-term patterns

**Calendar Effects (~20%)**:
- Weekend days (Saturday, Friday) show strong patterns
- Day-of-week features outweigh month-based seasonality in this dataset

**Promotional Indicators (~17%)**:
- Marketing campaigns and holidays have measurable impact
- Second-strongest individual feature after rolling averages

**Recent Lags (~8%)**:
- Direct lag features provide supplementary signal
- Less important than smoothed rolling averages

### Technical Insights

**Rolling Averages vs. Lags**:
- The 7-day rolling average has 12x higher SHAP importance than lag_1
- This suggests smoothed trends are more predictive than raw recent values
- Indicates the model benefits from noise reduction

**Calendar Patterns**:
- Day-of-week effects dominate over monthly patterns
- Saturday (21.1) and Friday (20.9) show strongest calendar features
- Suggests strong intra-week seasonality in the data

**Minimal Overfitting**:
- CV MAE (36.6) closely matches test MAE (38.6)
- Difference of only 2 units indicates good generalization
- Model is stable across different time periods

---

## Implementation Details

### Tech Stack

```python
# Core Libraries
pandas==2.x          # Data manipulation
numpy==1.x           # Numerical computing
scikit-learn==1.x    # ML utilities and validation

# Modeling
xgboost==2.x         # Gradient boosting

# Explainability
shap==0.42.x         # Model interpretation

# Visualization
matplotlib==3.x      # Plotting
seaborn==0.12.x      # Statistical visualization
```

### XGBoost Configuration

```python
XGBRegressor(
    n_estimators=600,       # Number of boosting rounds
    max_depth=8,            # Maximum tree depth
    learning_rate=0.05,     # Step size shrinkage
    subsample=0.8,          # Row sampling ratio
    colsample_bytree=0.8,   # Column sampling ratio
    random_state=42         # Reproducibility
)
```

### SHAP Implementation (Modern API)

```python
# Use modern SHAP API
explainer = shap.Explainer(model)
shap_values = explainer(X_test_sample)

# Global feature importance
shap.summary_plot(shap_values, X_test_sample, plot_type='bar')

# Individual prediction explanation
shap.waterfall_plot(shap_values[idx])
```

---

## Repository Structure

```
demand-forecasting-demo/
│
├── README.md                               # This file
├── Demand_Forecasting_Portfolio.ipynb     # Main analysis notebook
│
└── data/                                   # Data directory (not included)
    └── demand_forecasting_data.csv        # Sample dataset
```

---

## Running the Notebook

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

### Quick Start

1. **Prepare data**: Place `demand_forecasting_data.csv` in the working directory
2. **Launch Jupyter**: `jupyter notebook`
3. **Open**: `Demand_Forecasting_Portfolio.ipynb`
4. **Run all cells**: Cell → Run All

### Expected Runtime

- Full notebook execution: ~5-10 minutes
- SHAP calculation (1000 samples): ~2-3 minutes
- Reduce sample size in SHAP section if needed for faster execution

---

## Key Takeaways

### What This Project Demonstrates

✅ **Time-Series ML**: Proper handling of temporal data with lag features and rolling statistics

✅ **Feature Engineering**: Creating predictive features from raw time-series data

✅ **Model Comparison**: Systematic evaluation of multiple algorithms (Linear, RF, XGBoost)

✅ **Proper Validation**: Time-series cross-validation to prevent data leakage

✅ **Explainability**: SHAP analysis to interpret black-box model predictions

✅ **Code Quality**: Clean, documented, reproducible analysis

✅ **Communication**: Professional notebook with clear visualizations and explanations

### Technical Skills Showcased

- **Python**: pandas, numpy, scikit-learn, XGBoost
- **Machine Learning**: Regression, ensemble methods, gradient boosting
- **Time-Series**: Lag features, rolling windows, temporal validation
- **Model Evaluation**: Cross-validation, train/test splits, multiple metrics
- **Explainable AI**: SHAP values, feature importance, prediction attribution
- **Visualization**: matplotlib, seaborn, SHAP plots
- **Documentation**: Markdown, code comments, technical writing

---

## Model Performance Summary

**Final Model**: XGBoost Regressor

**Metrics**:
- Cross-Validation MAE: 36.6 units
- Test Set MAE: 38.6 units
- R² Score: 0.85
- RMSE: 52.3 units

**Error Distribution**:
- 50% of predictions within ±29 units
- 80% of predictions within ±68 units
- 95% of predictions within ±131 units

**Interpretation**: For products with ~350 unit average demand, this translates to approximately 10% average forecast error, which is reasonable for a demonstration model.

---

## Future Extensions

Potential enhancements for a production system:

**Technical Improvements**:
- Hyperparameter tuning (GridSearch/Optuna)
- Ensemble of multiple models
- Deep learning approaches (LSTM, Transformer)
- Multi-step ahead forecasting (predict next 7-30 days)

**Feature Engineering**:
- External data integration (weather forecasts, economic indicators)
- Product-specific features (category, price tier)
- Interaction features between promotions and pricing

**Deployment**:
- Model serving API (FastAPI/Flask)
- Automated retraining pipeline
- Monitoring and drift detection
- A/B testing framework

---

## Contact

For questions about this project or to discuss machine learning opportunities:

- **GitHub**: [Your GitHub Profile]
- **LinkedIn**: [Your LinkedIn]
- **Email**: [Your Email]

---

## Acknowledgments

- **XGBoost**: Distributed gradient boosting library by DMLC
- **SHAP**: Interpretability library by Scott Lundberg
- **scikit-learn**: Machine learning toolkit

---

*This is a portfolio project demonstrating machine learning and data science capabilities. The analysis showcases technical skills in time-series forecasting, model development, and explainable AI.*
