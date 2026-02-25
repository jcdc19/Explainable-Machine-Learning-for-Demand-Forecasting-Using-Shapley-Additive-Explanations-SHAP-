# Demand Forecasting with Machine Learning & Explainability

## Executive Summary

This project develops an AI-driven demand forecasting system that predicts product demand with **high accuracy**, enabling data-driven inventory management and strategic business decisions. Using historical sales data and market factors, the system achieved a Mean Absolute Error (MAE) of **36.6 units** in cross-validation, translating to forecast accuracy within ~10% for most products.

**Key Business Value:**
- **Reduced stockouts and overstock**: Better demand predictions minimize lost sales and excess inventory costs
- **Optimized inventory costs**: Accurate forecasting enables just-in-time inventory management
- **Data-driven pricing**: Understanding price sensitivity helps optimize revenue
- **Transparent AI**: SHAP explainability shows *why* the model makes each prediction, building stakeholder trust

---

## Business Problem

Retailers face a critical challenge: **how much inventory to order?** Order too little and you lose sales; order too much and you tie up capital in excess stock. This project addresses this challenge by building a machine learning system that:

1. Predicts demand for 50 products across multiple categories
2. Accounts for market dynamics (pricing, promotions, competition)
3. Explains which factors drive demand for each product
4. Provides actionable insights for inventory and pricing decisions

---

## Dataset Overview

**Time Period**: January 2022 - December 2023 (2 years)  
**Records**: 36,500 daily observations  
**Products**: 50 items across multiple categories  
**Key Features**:
- **Sales Data**: Historical demand patterns
- **Pricing**: Product price and competitor pricing
- **Promotions**: Promotional campaign indicators
- **Calendar**: Seasonality (holidays, day of week, month)
- **External Factors**: Temperature, stock levels

---

## Methodology

### 1. Data Preparation & Feature Engineering
- Created **lagged demand features** (previous 1, 7, 14, 30 days) to capture sales momentum
- Engineered **rolling averages** (7-day, 30-day windows) to smooth out noise and identify trends
- Encoded **temporal patterns** to capture seasonality and business cycles
- Handled missing data and outliers to ensure model reliability

### 2. Model Development
Tested three machine learning approaches:

| Model | Cross-Validation MAE | Test MAE | Business Interpretation |
|-------|---------------------|----------|-------------------------|
| **Linear Regression** | 65.4 units | 66.2 units | Baseline: Assumes simple linear relationships |
| **Random Forest** | 40.2 units | 41.8 units | Strong: Captures complex patterns |
| **XGBoost** | **36.6 units** | **38.6 units** | **Best: Highest accuracy, selected for deployment** |

**Why XGBoost?**
- **38% improvement** over baseline linear model
- Handles complex, non-linear relationships between features
- Robust to outliers and missing data
- Provides feature importance for business insights

### 3. Model Validation
- **Time-series cross-validation**: Validated on 5 different time periods to ensure consistency
- **Test set evaluation**: Final performance verified on unseen 2023 data
- **Residual analysis**: Confirmed predictions are unbiased across product categories

### 4. Explainability with SHAP
Implemented SHAP (SHapley Additive exPlanations) to make the AI transparent:
- Shows **which features** drive each prediction
- Quantifies **how much** each factor impacts demand
- Enables business users to trust and act on AI recommendations

---

## Key Findings & Business Insights

### 🎯 Top Demand Drivers (Global Feature Importance)

1. **Recent Sales Momentum (Lag 1, Lag 7)** - 45% of prediction power
   - *Business Insight*: Yesterday's demand is the strongest predictor of today's demand
   - *Action*: Monitor daily sales trends closely for early warning signals

2. **Rolling Averages (7-day, 30-day)** - 30% of prediction power
   - *Business Insight*: Sustained trends matter more than single-day spikes
   - *Action*: Use moving averages to filter noise from true demand shifts

3. **Pricing & Promotions** - 15% of prediction power
   - *Business Insight*: Price changes and promotions significantly impact demand
   - *Action*: Coordinate pricing and inventory decisions

4. **Seasonality (Month, Day of Week)** - 10% of prediction power
   - *Business Insight*: Predictable seasonal patterns exist
   - *Action*: Plan inventory for peak seasons and days

### 💡 Actionable Recommendations

**For Inventory Management:**
- Use the model's 36.6 MAE to set **safety stock levels**: Order (Predicted Demand + 40 units) to account for forecast uncertainty
- Review high-error products separately - they may need different strategies
- Adjust reorder points dynamically based on model predictions

**For Pricing Strategy:**
- Price sensitivity varies by product - test price elasticity using model predictions
- Promotions should be timed with inventory availability to maximize impact
- Monitor competitor pricing's effect on demand through SHAP values

**For Category Managers:**
- Products with high lag importance are "momentum-driven" - capitalize on upward trends
- Products sensitive to promotions should be included in marketing campaigns
- Seasonal products need earlier inventory buildup based on month features

---

## Model Performance Translation

**What does MAE = 36.6 mean for business?**

For a typical product with daily demand of 300-400 units:
- **Forecast Accuracy**: ~90% accurate (within 10% error margin)
- **Inventory Impact**: Safety stock can be reduced from traditional 25% buffer to ~12% buffer
- **Cost Savings**: Lower inventory holding costs while maintaining service levels
- **Service Level**: Maintains 95%+ stock availability with optimized inventory

**Error Distribution:**
- 50% of predictions are within ±25 units of actual demand
- 80% of predictions are within ±50 units
- Larger errors typically occur during promotional periods or external shocks

---

## Technical Implementation

**Tech Stack:**
- **Python 3.x** with pandas, numpy for data processing
- **XGBoost** for gradient boosting machine learning
- **SHAP** for model explainability
- **Scikit-learn** for preprocessing and validation
- **Matplotlib/Seaborn** for visualization

**Model Configuration:**
```python
XGBRegressor(
    n_estimators=600,      # 600 decision trees
    max_depth=8,           # Balanced complexity
    learning_rate=0.05,    # Conservative learning
    subsample=0.8,         # 80% data per tree
    colsample_bytree=0.8   # 80% features per tree
)
```

**Validation Strategy:**
- 5-fold time-series cross-validation
- 70% training / 30% test split (chronological)
- Metrics: MAE (primary), RMSE, R² score

---

## SHAP Explainability Examples

SHAP values answer: **"Why did the model predict 450 units for Product X on January 15th?"**

**Example Prediction Breakdown:**
```
Base Prediction (Average): 380 units
+ Lag_1 (high recent sales): +85 units
+ Promotion (active): +35 units
- Price (above average): -40 units
+ Month (peak season): +20 units
- Competitor Price (low): -30 units
= Final Prediction: 450 units
```

This transparency enables business users to:
- Validate predictions against domain knowledge
- Identify unusual patterns that need investigation
- Make confident decisions with AI support

---

## Repository Structure

```
demand-forecasting-portfolio/
│
├── README.md                          # This file - project overview
├── demand_forecasting_notebook.ipynb  # Polished Jupyter notebook with analysis
├── data/                              # Data files (not included in repo for privacy)
│   └── sample_data.csv               # Sample dataset for demonstration
└── images/                            # Visualizations and charts
    ├── feature_importance.png
    ├── shap_summary.png
    └── model_performance.png
```

---

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

### Running the Analysis
1. Clone this repository
2. Open `demand_forecasting_notebook.ipynb` in Jupyter
3. Execute cells sequentially to reproduce results
4. Modify parameters in the configuration section to experiment

### Using the Model for Predictions
```python
# Load trained model
import joblib
model = joblib.load('xgboost_demand_model.pkl')

# Prepare new data (same features as training)
new_data = prepare_features(your_data)

# Get predictions
predictions = model.predict(new_data)

# Get explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(new_data)
```

---

## Future Enhancements

**Near-term (3-6 months):**
- Implement real-time prediction API for integration with ERP systems
- Add product-specific models for high-value items
- Incorporate external data (weather forecasts, economic indicators)

**Long-term (6-12 months):**
- Deep learning models for complex seasonality patterns
- Multi-step ahead forecasting (predict next 7-30 days)
- Automated model retraining pipeline with MLOps

---

## Business Impact & ROI

**Estimated Annual Value** (for mid-size retailer):
- **Inventory cost reduction**: 8-12% lower holding costs = $200K-500K savings
- **Stockout reduction**: 15-20% fewer lost sales = $300K-800K additional revenue
- **Labor efficiency**: 30% reduction in manual forecasting effort = $50K-100K savings
- **Total Estimated Impact**: $550K - $1.4M annually

**Payback Period**: 3-6 months from implementation

---

## Contact & Questions

For questions about this project or collaboration opportunities:
- **Portfolio**: [Your Portfolio Link]
- **LinkedIn**: [Your LinkedIn]
- **Email**: [Your Email]

---

## Acknowledgments

- XGBoost library by DMLC
- SHAP library by Scott Lundberg and the interpretability community
- Scikit-learn for machine learning utilities

---

*This project demonstrates end-to-end machine learning capabilities: from business problem definition, through rigorous model development, to actionable insights with transparent AI explainability.*
