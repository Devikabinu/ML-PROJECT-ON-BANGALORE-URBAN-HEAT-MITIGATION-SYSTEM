import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64

st.set_page_config(page_title="Bangalore Temperature Predictor", layout="wide")

st.title("🌡️ Bangalore Temperature Prediction Dashboard")

# Upload dataset
st.sidebar.header("Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df.dropna()

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.info("No file uploaded. Using default dataset.")
    data = load_data("bangalore_uhi_dataset_with_names.csv")

# Features and Target
features = ["green_cover_pct", "building_density", "proximity_to_water", "population_density"]
target = "avg_temp"
X = data[features]
y = data[target]

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X, y)
preds = model.predict(X)

# Add predictions to DataFrame
data["Predicted"] = preds
data["Absolute_Error"] = abs(data["Predicted"] - y)
data["Percent_Error"] = (data["Absolute_Error"] / y) * 100

# Metrics
mae = mean_absolute_error(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
r2 = r2_score(y, preds)

st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.3f}")

# Top 10 hottest
top_10 = data.sort_values(by="Predicted", ascending=False).head(10)

st.subheader("🔥 Top 10 Predicted Hottest Regions")
st.dataframe(top_10[["region_name", "Predicted", "Absolute_Error", "Percent_Error"]])

# Plot 1: Top 10 hottest bar chart
st.subheader("📈 Hottest Regions Bar Chart")
fig1, ax1 = plt.subplots()
sns.barplot(x="Predicted", y="region_name", data=top_10, palette="Reds_d", ax=ax1)
ax1.set_xlabel("Predicted Temperature (°C)")
ax1.set_ylabel("Region Name")
st.pyplot(fig1)

# Plot 2: Actual vs Predicted
st.subheader("⚖️ Actual vs Predicted Temperature")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=y, y=preds, alpha=0.6, color="teal", ax=ax2)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_xlabel("Actual Temp (°C)")
ax2.set_ylabel("Predicted Temp (°C)")
st.pyplot(fig2)

# Plot 3: Feature importance
st.subheader("🧠 Feature Importance")
importances = model.feature_importances_
fig3, ax3 = plt.subplots()
sns.barplot(x=importances, y=features, color="skyblue", ax=ax3)
st.pyplot(fig3)

# Download button
st.subheader("📁 Download Prediction Results")
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download CSV</a>'
    return href

st.markdown(get_table_download_link(data[["region_id", "region_name", target, "Predicted", "Absolute_Error", "Percent_Error"]]), unsafe_allow_html=True)
# Best places to migrate (coolest regions)
st.subheader("❄️ Best Places to Migrate for Lower Temperatures")

# Sort by lowest predicted temperature
coolest_regions = data.sort_values(by="Predicted", ascending=True).head(10)

st.write("Here are the top 10 coolest predicted regions in Bangalore:")

st.dataframe(coolest_regions[["region_name", "Predicted", "Absolute_Error", "Percent_Error"]])

# Plot: Coolest regions
fig_cool, ax_cool = plt.subplots()
sns.barplot(x="Predicted", y="region_name", data=coolest_regions, palette="Blues_d", ax=ax_cool)
ax_cool.set_title("Top 10 Coolest Predicted Regions in Bangalore")
ax_cool.set_xlabel("Predicted Temperature (°C)")
ax_cool.set_ylabel("Region Name")
st.pyplot(fig_cool)
