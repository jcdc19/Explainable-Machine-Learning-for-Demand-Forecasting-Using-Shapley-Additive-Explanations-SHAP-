# Setup Guide - Demand Forecasting Portfolio Notebook

## Quick Start

### 1. Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

### 2. Prepare Your Data
Place your `demand_forecasting_data.csv` file in the same directory as the notebook. The CSV should have these columns:
- `date`, `product_id`, `category`, `demand`, `price`, `has_promotion`, `is_holiday`, `day_of_week`, `month`, `temperature`, `competitor_price`, `stock_level`

### 3. Run the Notebook
```bash
jupyter notebook Demand_Forecasting_Portfolio.ipynb
```

Execute cells sequentially from top to bottom.

---

## Troubleshooting SHAP Visualizations

### Issue: Empty SHAP Plots

**Cause**: SHAP plots only appear after code execution with actual data.

**Solution**:
1. Ensure all previous cells have been executed successfully
2. Verify the model is trained (check for `xgb_model` variable)
3. Confirm `X_test` has data (run `print(X_test.shape)`)
4. Run the SHAP calculation cell:

```python
# This cell must complete before plots appear
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
print("✓ SHAP values calculated successfully")
print(f"Shape: {shap_values.shape}")
```

### Common SHAP Errors & Fixes

**Error: "NameError: name 'xgb_model' is not defined"**
- **Fix**: Run the model training cells first (Section 4)

**Error: "SHAP plots are blank/empty"**
- **Fix**: Ensure you're running in Jupyter (not just viewing the .ipynb file)
- **Fix**: Run `%matplotlib inline` at the start of the notebook
- **Fix**: Use `plt.show()` after each SHAP plot

**Error: SHAP calculation takes too long**
- **Fix**: Use a sample of test data:
```python
# Sample 1000 rows for faster SHAP calculation
sample_size = min(1000, len(X_test))
X_test_sample = X_test.iloc[:sample_size]
shap_values_sample = explainer.shap_values(X_test_sample)
```

### Enhanced SHAP Code (with error handling)

Replace the SHAP section with this more robust version:

```python
import shap
import matplotlib.pyplot as plt

# Ensure inline plotting
%matplotlib inline

print("Step 1: Creating SHAP explainer...")
explainer = shap.TreeExplainer(xgb_model)

print("Step 2: Calculating SHAP values...")
# Use sample for faster computation (optional)
sample_size = min(1000, len(X_test))
X_test_sample = X_test.iloc[:sample_size]
shap_values = explainer.shap_values(X_test_sample)

print(f"✓ SHAP values calculated: {shap_values.shape}")

print("Step 3: Generating global feature importance plot...")
# Global feature importance (bar plot)
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type='bar', max_display=15)
plt.title('Global Feature Importance (SHAP)', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("Step 4: Generating detailed SHAP summary plot...")
# Detailed summary plot (beeswarm)
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, max_display=15)
plt.tight_layout()
plt.show()

print("Step 5: Generating waterfall plot for single prediction...")
# Single prediction explanation
sample_idx = 0
fig = plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap.Explanation(
    values=shap_values[sample_idx],
    base_values=explainer.expected_value,
    data=X_test_sample.iloc[sample_idx],
    feature_names=X_test_sample.columns.tolist()
))
plt.tight_layout()
plt.show()

print("✓ All SHAP visualizations complete!")
```

---

## Expected Output Examples

When the notebook runs successfully, you should see:

### 1. Global Feature Importance (Bar Chart)
- Horizontal bars showing top 15 features
- X-axis: Mean absolute SHAP value
- Bars ordered from most to least important
- Typically shows lag features at the top

### 2. SHAP Summary Plot (Beeswarm)
- Each dot represents one prediction
- X-axis: SHAP value (impact on prediction)
- Color: Feature value (red=high, blue=low)
- Features ordered by importance (top to bottom)

### 3. Waterfall Plot
- Shows single prediction breakdown
- Starts from base value (average demand)
- Each bar shows one feature's contribution
- Red bars push prediction higher, blue bars lower

---

## Performance Tips

### Speed Up SHAP Calculations
```python
# Use tree_path_dependent=False for faster but approximate SHAP values
explainer = shap.TreeExplainer(xgb_model, tree_path_dependent=False)

# Or use a smaller sample
shap_values = explainer.shap_values(X_test.iloc[:500])
```

### Memory Issues
If you run out of memory:
```python
# Calculate SHAP values in batches
batch_size = 200
all_shap_values = []

for i in range(0, len(X_test), batch_size):
    batch = X_test.iloc[i:i+batch_size]
    batch_shap = explainer.shap_values(batch)
    all_shap_values.append(batch_shap)
    print(f"Processed {min(i+batch_size, len(X_test))}/{len(X_test)} rows")

shap_values = np.vstack(all_shap_values)
```

---

## Saving SHAP Plots

To save visualizations for your portfolio:

```python
# Save global feature importance
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type='bar', max_display=15, show=False)
plt.title('Global Feature Importance (SHAP)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save summary plot
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ SHAP plots saved to current directory")
```

---

## Verifying Your Setup

Run this diagnostic code to verify everything is working:

```python
# Diagnostic script
print("=== Demand Forecasting Setup Diagnostics ===\n")

# 1. Check libraries
try:
    import pandas as pd
    import numpy as np
    import xgboost
    import shap
    import matplotlib.pyplot as plt
    print("✓ All libraries imported successfully")
except ImportError as e:
    print(f"✗ Missing library: {e}")

# 2. Check data
try:
    df = pd.read_csv('demand_forecasting_data.csv')
    print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Data file not found: demand_forecasting_data.csv")

# 3. Check model (after training)
try:
    print(f"✓ Model exists: {type(xgb_model)}")
    print(f"✓ Test set size: {X_test.shape}")
except NameError:
    print("✗ Model not trained yet - run training cells first")

# 4. Test SHAP
try:
    test_explainer = shap.TreeExplainer(xgb_model)
    test_shap = test_explainer.shap_values(X_test.iloc[:10])
    print(f"✓ SHAP working correctly: {test_shap.shape}")
except Exception as e:
    print(f"✗ SHAP error: {e}")

print("\n=== Diagnostics Complete ===")
```

---

## Next Steps After Successful Execution

1. **Review outputs** - Check all visualizations rendered correctly
2. **Save key plots** - Export SHAP plots for presentations
3. **Document findings** - Note the top feature importances for your README
4. **Export model** - Save trained model for deployment:
   ```python
   import joblib
   joblib.dump(xgb_model, 'xgboost_demand_model.pkl')
   ```

5. **Create portfolio images folder**:
   ```bash
   mkdir images
   mv shap_*.png images/
   ```

---

## Getting Help

If you encounter issues not covered here:
1. Check the [SHAP documentation](https://shap.readthedocs.io/)
2. Verify XGBoost version compatibility: `pip install xgboost==1.7.0`
3. Ensure Jupyter is up to date: `pip install --upgrade jupyter`

Common version combinations that work well:
- Python 3.8+
- XGBoost 1.7.0
- SHAP 0.41.0+
- scikit-learn 1.0+
