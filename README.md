# Demand Forecasting & Inventory Optimization

An end-to-end Data Science project that predicts future product demand and helps optimize inventory decisions using time series forecasting and an interactive Streamlit dashboard.

## Live Demo
https://demand-forecasting-inventory-optimization.streamlit.app/

---

## Project Overview

Efficient inventory management is critical for businesses to avoid:

- Stockouts (lost sales)
- Overstocking (high holding cost)

This project solves the problem using:

✔ Time Series Forecasting (SARIMAX)  
✔ Demand Prediction  
✔ Interactive Dashboard (Streamlit)  

---

## Key Features

- Demand forecasting using SARIMAX model
- Interactive visualization with Plotly
- Real-time predictions via Streamlit app
- Inventory decision support
- Product-level forecasting capability

---

##  Tech Stack

- **Python**
- **Pandas, NumPy**
- **Statsmodels (SARIMAX)**
- **Plotly**
- **Streamlit**
- **Matplotlib**
- **Joblib**

---

## Project Structure

Demand-Forecasting/
│
├── app.py # Streamlit frontend
├── backend.py # Prediction logic
├── demand_forecasting_model.pkl # Trained SARIMAX model
├── demand_inventory.csv 
├── requirements.txt 
└── README.md


---

##  How It Works

1. Load historical demand data
2. Train SARIMAX model
3. Save model using Joblib
4. Use Streamlit app for:
   - Selecting product
   - Generating demand forecasts
5. Display results using charts

---

##  Model Details

- Model Used: **SARIMAX (Seasonal ARIMA with exogenous variables)**
- Captures:
  - Trend 
  - Seasonality 
  - Noise 

---

##  Output

- Future demand predictions
- Time series visualization
- Forecast trends

---
