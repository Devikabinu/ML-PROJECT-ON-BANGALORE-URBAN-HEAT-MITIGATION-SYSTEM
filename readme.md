# 🌆 ML Project: Bangalore Urban Heat Mitigation System

An interactive **Streamlit dashboard** that predicts and visualizes urban temperature patterns across Bangalore using Machine Learning. This tool aids in identifying heat-prone and cooler regions to support urban heat mitigation efforts.

---

## 🚀 Features

- 📁 Upload your own CSV dataset or use the built-in default dataset.
- 🔍 Uses a **Random Forest Regressor** to predict average temperatures.
- 📊 Visual insights:
  - **Model Performance** (MAE, RMSE, R² Score)
  - **Top 10 Hottest & Coolest Regions**
  - **Actual vs Predicted Scatter Plot**
  - **Feature Importance Analysis**
- 📥 Download the prediction results as a CSV file.

---

## 📊 Input Dataset Requirements

Your dataset should contain the following columns:

- `region_id`
- `region_name`
- `green_cover_pct`
- `building_density`
- `proximity_to_water`
- `population_density`
- `avg_temp` (target column)

If no file is uploaded, the app uses a default dataset: `bangalore_uhi_dataset_with_names.csv`.

---

## 🖼️ Screenshots

### 🔥 Hottest Regions
![Hottest Regions](screenshots/hottest_regions.png)

### ❄️ Coolest Regions
![Coolest Regions](screenshots/coolest_regions.png)

### 📈 Actual vs Predicted
![Actual vs Predicted](s)
