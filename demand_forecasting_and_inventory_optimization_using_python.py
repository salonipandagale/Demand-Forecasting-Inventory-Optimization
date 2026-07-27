# -*- coding: utf-8 -*-
"""Demand Forecasting and Inventory Optimization using Python.ipynb
"""

import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv("demand_inventory.csv")
print(data.head())

data = data.drop(columns=['Unnamed: 0'])

fig_demand = px.line(data, x='Date',
                     y='Demand',
                     title='Demand Over Time')
fig_demand.show()

"""* We can see seasonal patterns in the demand. We can forecast the demand using SARIMA"""

fig_inventory = px.line(data, x='Date',
                        y='Inventory',
                        title='Inventory Over Time')
fig_inventory.show()

data['Date'] = pd.to_datetime(data['Date'])
time_series = data.set_index('Date')['Demand']

differenced_series = time_series.diff().dropna()

# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

"""The value of p, d, and q is 1, 1, 1 here

## Demand Forecasting
"""

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 2) #2 because the data contains a time period of 2 months only
model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

import joblib
joblib.dump(model_fit, "demand_forecasting_model.pkl")

future_steps = 10
predictions = model_fit.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)
print(predictions)

"""## Inventory Optimization"""

future_dates = pd.date_range(start=time_series.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')

forecasted_demand = pd.Series(predictions, index=future_dates)

initial_inventory = 5500

lead_time = 1 # it's different for every business, 1 is an example

service_level = 0.95

z = np.abs(np.percentile(forecasted_demand, 100 * (1 - service_level)))
order_quantity = np.ceil(forecasted_demand.mean() + z).astype(int)

reorder_point = forecasted_demand.mean() * lead_time + z

safety_stock = reorder_point - forecasted_demand.mean() * lead_time

holding_cost = 0.1
stockout_cost = 10
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, forecasted_demand.mean() * lead_time - initial_inventory)

# Calculate the total cost
total_cost = total_holding_cost + total_stockout_cost

print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)

"""1. Optimal Order Quantity: 236 – The optimal order quantity refers to the quantity of a product that should be ordered from suppliers when the inventory level reaches a certain point.

2. Reorder Point: 235.25 – The reorder point is the inventory level at which a new order should be placed to replenish stock before it runs out.

3. Safety Stock: 114.45 – Safety stock is the additional inventory kept on hand to account for uncertainties in demand and supply.

4. It acts as a buffer against unexpected variations in demand or lead time.

5. Total Cost: 561.80 – The total cost represents the combined costs associated with inventory management.

## Conclusion
* Demand Forecasting involves predicting the quantity and pattern of customer orders, which is crucial for businesses to efficiently allocate resources, manage inventory, and plan production.
* Inventory Optimization aims to strike a balance between having sufficient stock to meet demand without carrying excess inventory that ties up capital and storage space.
"""