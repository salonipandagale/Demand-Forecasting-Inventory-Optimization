# Demand Forecasting and Inventory Optimization

A time-series forecasting and inventory planning system that combines SARIMA/SARIMAX demand forecasting with inventory optimization techniques such as safety stock, reorder point, Economic Order Quantity (EOQ), and inventory simulation.

The project uses the M5 retail sales dataset to forecast daily demand for a selected product-store combination and translates the forecast into practical inventory replenishment decisions.

---

## Live Demo

**Live Streamlit Application:**

https://demand-forecasting-inventory-optimization-bd9h.onrender.com/

---

## Project Overview

Accurate demand forecasting is an important component of inventory planning. Poor forecasts can lead to overstocking, unnecessary holding costs, or stockouts.

This project develops an end-to-end demand forecasting and inventory optimization pipeline:

```text
M5 Retail Sales Data
        |
        v
Data Preprocessing
        |
        v
Exploratory Data Analysis
        |
        v
Trend and Seasonality Analysis
        |
        v
ADF + ACF + PACF Analysis
        |
        v
Chronological Train / Validation / Test Split
        |
        v
SARIMA Modeling
        |
        v
SARIMAX Modeling with Exogenous Variables
        |
        v
Model Comparison
        |
        v
Final SARIMA Model
        |
        v
28-Day Demand Forecast
        |
        v
Prediction Intervals
        |
        v
Inventory Optimization
        |
        +------------------+
        |                  |
        v                  v
Safety Stock             EOQ
        |                  |
        +--------+---------+
                 |
                 v
          Reorder Point
                 |
                 v
       Inventory Simulation
                 |
                 v
       Streamlit Dashboard
```

The objective is not only to forecast demand, but also to convert the forecast into an inventory replenishment policy.

---

# Objectives

The main objectives of this project are:

1. Analyze historical retail demand patterns.
2. Identify trend and weekly seasonal behavior.
3. Test stationarity using the Augmented Dickey-Fuller test.
4. Use ACF and PACF to guide SARIMA model selection.
5. Develop multiple SARIMA specifications.
6. Evaluate SARIMAX models using price and event information as exogenous variables.
7. Compare SARIMA and SARIMAX using chronological validation data.
8. Select the forecasting model based on out-of-sample performance.
9. Generate a 28-day future demand forecast with prediction intervals.
10. Convert demand forecasts into inventory decisions.
11. Calculate safety stock and reorder point for different service levels.
12. Calculate Economic Order Quantity (EOQ).
13. Simulate inventory replenishment over the future evaluation period.
14. Analyze stockouts, fill rate, number of orders, and inventory cost.
15. Deploy the complete system using Streamlit.

---

# Dataset

This project uses the **M5 Forecasting dataset**, which contains retail sales information from Walmart stores.

The dataset contains information about:

* Products
* Departments
* Categories
* Stores
* States
* Daily sales
* Calendar information
* Events
* Product selling prices

The main files used are:

```text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
```

The original sales data is provided in wide format, where each daily observation is represented by columns such as:

```text
d_1
d_2
d_3
...
d_1913
```

The project transforms the sales data into a long time-series format and combines it with calendar and pricing information.

---

# Dataset Exploration

Initial dataset sizes:

| Dataset          |          Shape |
| ---------------- | -------------: |
| Calendar         |     1,969 × 14 |
| Sales Training   | 30,490 × 1,919 |
| Sell Prices      |  6,841,121 × 4 |
| Sales Evaluation | 30,490 × 1,947 |

After converting the sales data from wide format to long format, the resulting dataset contains more than 58 million daily product-store observations.

The dataset contains:

* 10 stores
* 3 states
* 3 product categories
* 1,913 historical daily observations for the selected series

---

# Selected Product and Store

Instead of modeling all 30,490 product-store combinations simultaneously, the project focuses on one representative product-store time series.

Selected product:

```text
Item: FOODS_3_586
Store: TX_2
```

The selected series contains:

```text
Historical observations: 1,913
Start date: 2011-01-29
End date: 2016-04-24
Average daily demand: approximately 100.80 units
```

The series was selected based on demand characteristics including sufficient demand volume, relatively low intermittency, and meaningful demand variability.

The selected product has approximately:

```text
Mean demand: 100.80
Standard deviation: 30.88
Coefficient of variation: 0.31
Zero-demand proportion: approximately 0.42%
```

This makes the series suitable for classical time-series forecasting.

---

# Data Preprocessing

The original M5 sales data is stored in wide format.

The daily sales columns were identified using:

```python
demand_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]
```

The data was then transformed using a wide-to-long operation.

The resulting structure contains:

```text
date
item_id
store_id
Demand
wm_yr_wk
weekday
month
year
event_name_1
event_type_1
```

The transformed sales data was then merged with the calendar dataset.

Price information was merged using:

```text
store_id
item_id
wm_yr_wk
```

This allowed the project to use:

* Historical demand
* Selling price
* Calendar information
* Event information

for the forecasting experiments.

---

# Exploratory Data Analysis

The demand series was analyzed to understand its temporal structure.

The analysis included:

* Historical demand visualization
* Average demand
* Demand variability
* Weekly demand patterns
* Rolling statistics
* ADF stationarity test
* ACF analysis
* PACF analysis

---

# Weekly Seasonality

The average demand by weekday showed a clear weekly pattern.

Approximate average demand:

| Day       | Average Demand |
| --------- | -------------: |
| Monday    |          88.18 |
| Tuesday   |          84.29 |
| Wednesday |          85.09 |
| Thursday  |          91.48 |
| Friday    |         104.84 |
| Saturday  |         128.13 |
| Sunday    |         123.43 |

Demand increases significantly toward the weekend.

This provided evidence for a weekly seasonal period:

```text
s = 7
```

The ACF also showed strong spikes around:

```text
7
14
21
28
35
```

which further supported weekly seasonality.

---

# Stationarity Analysis

The Augmented Dickey-Fuller test was performed on the training series.

The original demand series produced:

```text
ADF statistic: -3.7950
p-value: 0.00296
```

Since the p-value is below 0.05, the null hypothesis of a unit root was rejected.

Therefore, the original training series was considered stationary at the 5% significance level.

As a result, the initial SARIMA models used:

```text
d = 0
```

First differencing was also tested, but the original series already passed the stationarity test, so differencing was not required for the selected model.

---

# ACF and PACF Analysis

The ACF showed strong short-term dependence and strong seasonal correlation.

Important ACF values included:

```text
Lag 1  : approximately 0.54
Lag 7  : approximately 0.58
Lag 14 : approximately 0.52
Lag 21 : approximately 0.52
Lag 28 : approximately 0.49
```

The PACF showed a strong initial spike at lag 1.

These patterns motivated testing models with:

```text
p ≈ 1–2
q ≈ 1
P = 1
Q = 1
s = 7
```

Rather than relying only on automated model selection, multiple candidate specifications were evaluated using chronological validation.

---

# Train, Validation and Test Split

Because this is a time-series problem, the data was split chronologically rather than randomly.

The split was approximately:

```text
70% Training
15% Validation
15% Test
```

This prevents future observations from being used to train the model and better reflects real-world forecasting.

The training period ends around:

```text
2014-09-28
```

The test period covers approximately:

```text
2015-07-13 to 2016-04-24
```

---

# SARIMA Modeling

The project uses the SARIMA framework:

```text
SARIMA(p,d,q)(P,D,Q,s)
```

where:

* `p` = autoregressive order
* `d` = non-seasonal differencing
* `q` = moving-average order
* `P` = seasonal autoregressive order
* `D` = seasonal differencing
* `Q` = seasonal moving-average order
* `s` = seasonal period

Because weekly seasonality was identified:

```text
s = 7
```

---

# SARIMA Model Experiments

Multiple SARIMA specifications were evaluated.

The strongest candidate was:

```text
SARIMA(2,0,1)(1,0,1,7)
```

Validation performance:

| Metric |     Value |
| ------ | --------: |
| MAE    |     16.03 |
| RMSE   |     21.24 |
| MAPE   |    45.98% |
| AIC    | 11,883.23 |
| BIC    | 11,914.39 |

The model achieved the lowest validation MAE and RMSE among the tested SARIMA candidates.

A model with seasonal differencing was also tested:

```text
SARIMA(1,0,1)(1,1,1,7)
```

However, its validation performance was worse:

```text
MAE: 18.14
RMSE: 23.16
MAPE: 51.37%
```

Therefore, seasonal differencing was not retained.

---

# Residual Diagnostics

After selecting the SARIMA model, residual diagnostics were performed.

The Ljung-Box test was evaluated at several lags.

Results:

| Lag | p-value |
| --: | ------: |
|   7 |   0.482 |
|  14 |   0.276 |
|  21 |   0.160 |
|  28 |   0.062 |

All tested p-values were above 0.05.

Therefore, at the 5% significance level, the Ljung-Box test did not detect statistically significant residual autocorrelation at the tested lags.

The lag-28 result was relatively close to 0.05, so the residuals should not be described as perfectly independent.

---

# SARIMAX Modeling

SARIMAX was also tested to determine whether external variables could improve forecasting performance.

Two types of exogenous information were considered:

```text
1. Selling price
2. Event indicator
```

The event indicator was created as:

```python
event_flag = event_name_1.notna().astype(int)
```

The selected product-store series contained:

```text
1,759 non-event days
154 event days
```

---

# SARIMA vs SARIMAX

Several SARIMAX specifications were evaluated.

| Model                   | Exogenous Variables | Validation MAE | Validation RMSE | Validation MAPE |
| ----------------------- | ------------------- | -------------: | --------------: | --------------: |
| SARIMA(2,0,1)(1,0,1,7)  | None                |          16.03 |           21.24 |          45.98% |
| SARIMAX(2,0,1)(1,0,1,7) | Price + Event       |          22.35 |           27.01 |          60.48% |
| SARIMAX(2,0,1)(1,0,1,7) | Price               |          22.19 |           26.84 |          60.29% |
| SARIMAX(1,0,1)(1,0,1,7) | Price + Event       |          21.42 |           26.13 |          58.44% |

The SARIMA model outperformed the tested SARIMAX models on the chronological validation set.

Therefore, the final forecasting model was selected based on out-of-sample performance rather than automatically preferring the more complex model.

This is an important result of the project: adding external variables does not necessarily improve forecasting accuracy.

---

# Final Forecasting Model

The final selected model is:

```text
SARIMA(2,0,1)(1,0,1,7)
```

After model selection, the model was refitted using the complete historical series of 1,913 observations.

A 28-day forecast was then generated.

The forecasting system also produces a 95% prediction interval to represent forecast uncertainty.

---

# Forecast Evaluation

The model was evaluated on both the chronological test period and a 28-day future holdout.

## Test Set Performance

```text
MAE  = 16.34
RMSE = 21.40
```

## 28-Day Future Holdout Performance

```text
MAE  = 11.40
RMSE = 14.08
MAPE = 15.54%
```

The 28-day evaluation demonstrates that the final model was able to capture the demand pattern reasonably well for the selected product-store combination.

MAE and RMSE are treated as the primary evaluation metrics, while MAPE is interpreted with caution because percentage-based errors can become unstable when actual demand values are small or zero.

---

# Inventory Optimization

The project extends demand forecasting into inventory planning.

The main inventory decisions are:

1. Safety Stock
2. Reorder Point
3. Economic Order Quantity
4. Inventory simulation

---

# Safety Stock

A target service level is used to determine the required safety stock.

The project uses the forecast error distribution to estimate demand uncertainty during the lead time.

For the baseline scenario:

```text
Service level = 95%
Lead time = 7 days
```

The residual standard deviation from the final SARIMA model was approximately:

```text
20.43 units
```

The estimated lead-time demand uncertainty was calculated using:

```text
Lead-time standard deviation
≈ Daily residual standard deviation × sqrt(Lead Time)
```

For a 7-day lead time:

```text
Lead-time demand standard deviation ≈ 54.05 units
```

At a 95% service level:

```text
Safety Stock ≈ 88.9 units
```

---

# Reorder Point

The reorder point combines expected lead-time demand and safety stock.

For the baseline 7-day lead time and 95% service level:

```text
Expected lead-time demand ≈ 522 units

Safety stock ≈ 89 units

Reorder point ≈ 611 units
```

Therefore:

```text
Reorder Point ≈ 611 units
```

The interpretation is:

> When inventory position falls to approximately 611 units, replenishment should be triggered.

The Streamlit dashboard allows the user to change service level and lead time interactively, so the reorder point changes accordingly.

---

# Economic Order Quantity

The project uses the Economic Order Quantity framework to determine an approximate replenishment quantity.

The annualized demand is approximately:

```text
36,793 units/year
```

The project assumes:

```text
Ordering cost = 50 per order
Holding cost = 2 per unit per year
```

Under these assumptions:

```text
EOQ ≈ 1,356 units
```

Therefore, the baseline inventory policy is approximately:

```text
Reorder Point ≈ 611 units
Order Quantity ≈ 1,356 units
```

---

# Inventory Simulation

The inventory policy was simulated using the 28-day future demand period.

The baseline simulation produced:

| Metric               |                  Result |
| -------------------- | ----------------------: |
| Total Demand         |             2,285 units |
| Units Sold           |             2,285 units |
| Stockout Units       |                       0 |
| Stockout Days        |                       0 |
| Fill Rate            |                    100% |
| Number of Orders     |                       2 |
| Average Inventory    | approximately 663 units |
| Holding Cost         |    approximately 101.73 |
| Ordering Cost        |                     100 |
| Stockout Cost        |                       0 |
| Total Inventory Cost |    approximately 201.73 |

The policy achieved a 100% fill rate during this particular 28-day evaluation period.

This result should be interpreted as the performance of the policy over the evaluated period rather than a guarantee of 100% service under all future conditions.

---

# Service-Level Analysis

The project also evaluates how changing the target service level affects inventory requirements.

Three service levels were considered:

```text
90%
95%
99%
```

Baseline results:

| Service Level | Safety Stock | Reorder Point |
| ------------: | -----------: | ------------: |
|           90% |        69.26 |        591.38 |
|           95% |        88.90 |        611.02 |
|           99% |       125.73 |        647.85 |

As the target service level increases:

```text
Service Level
      ↓
Higher Safety Stock
      ↓
Higher Reorder Point
      ↓
Higher Inventory Requirement
```

This demonstrates the trade-off between inventory availability and inventory carrying requirements.

---

# Important Business Assumptions

The M5 dataset does not provide actual inventory policy parameters such as:

* Lead time
* Ordering cost
* Holding cost
* Stockout cost
* Target service level

Therefore, these values were introduced as explicit business assumptions for the inventory optimization component.

The baseline assumptions are:

| Parameter     |      Assumption |
| ------------- | --------------: |
| Lead Time     |          7 days |
| Service Level |             95% |
| Ordering Cost |    50 per order |
| Holding Cost  | 2 per unit/year |
| Stockout Cost |      5 per unit |

These assumptions are used for demonstrating the inventory decision-making framework and should not be interpreted as values directly provided by the M5 dataset.

---

# Inventory Uncertainty Assumption

The lead-time demand standard deviation is approximated using the residual standard deviation of the forecasting model.

This approach assumes approximately independent daily forecast errors when scaling uncertainty with the square root of lead time.

Because time-series demand can contain temporal dependence, this is an approximation.

A more advanced implementation could estimate lead-time demand uncertainty using:

* Forecast-error simulation
* Rolling-origin forecasting
* Bootstrap methods
* Monte Carlo inventory simulation
* Full probabilistic forecasting

---

# Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard contains the following sections:

## 1. Demand Forecast

Displays:

* Historical demand
* Forecast demand
* 95% prediction interval
* Actual future demand
* Forecast table

## 2. Forecast Performance

Displays:

* MAE
* RMSE
* MAPE
* SARIMA vs SARIMAX comparison

## 3. Inventory Optimization

Allows users to change:

* Target service level
* Lead time
* Forecast horizon

The dashboard dynamically calculates:

* Lead-time demand
* Safety stock
* Reorder point
* EOQ

## 4. Inventory Simulation

Displays:

* Inventory trajectory
* Reorder points
* Replenishment events
* Stockout units
* Fill rate
* Number of orders
* Total inventory cost

## 5. Service-Level Scenario Analysis

Compares:

* 90% service level
* 95% service level
* 99% service level

and visualizes the effect on:

* Safety stock
* Reorder point

---

# Project Structure

```text
Demand-Forecasting-Inventory-Optimization/
│
├── data/
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   ├── sales_train_evaluation.csv
│   └── sell_prices.csv
│
├── notebooks/
│   └── demand_forecasting.ipynb
│
├── models/
│   ├── final_sarima_model.pkl
│   ├── historical_demand.pkl
│   ├── inventory_parameters.pkl
│   ├── model_comparison.pkl
│   └── future_forecast_results.pkl
│
├── app.py
│
└── README.md
```

---

# Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Visualization

* Matplotlib

## Time-Series Forecasting

* Statsmodels
* SARIMA
* SARIMAX
* ADF Test
* ACF
* PACF
* Ljung-Box Test

## Model Persistence

* Joblib

## Dashboard and Deployment

* Streamlit

---

# Installation

Clone the repository:

```bash
git clone <https://github.com/salonipandagale/Demand-Forecasting-Inventory-Optimization>
cd Demand-Forecasting-Inventory-Optimization
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib statsmodels joblib streamlit
```

---

# Running the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

The dashboard will open in the browser.

For the deployed version, use the live demo link provided at the top of this README.

---

# Model Deployment Approach

The trained SARIMA model and other required artifacts are saved using Joblib.

The Streamlit application loads these saved artifacts instead of retraining the forecasting model every time the application starts.

The saved artifacts include:

```text
final_sarima_model.pkl
historical_demand.pkl
inventory_parameters.pkl
model_comparison.pkl
future_forecast_results.pkl
```

This makes the deployed dashboard faster and avoids repeating the computationally expensive training process.

---

# Key Results

The final system achieved:

```text
Final Model:
SARIMA(2,0,1)(1,0,1,7)

Historical Observations:
1,913

Average Daily Demand:
100.80 units

Test MAE:
16.34 units

Test RMSE:
21.40 units

28-Day Future MAE:
11.40 units

28-Day Future RMSE:
14.08 units

28-Day Future MAPE:
15.54%

Baseline Safety Stock:
88.90 units

Baseline Reorder Point:
611.02 units

EOQ:
1,356.33 units

28-Day Fill Rate:
100%

Stockout Units:
0

Orders:
2

Total Modeled Inventory Cost:
approximately 201.73
```

---

# Key Insights

### 1. Weekly seasonality was important

Demand was substantially higher on weekends, and the ACF showed strong correlations at multiples of seven. This motivated a seasonal period of seven days.

### 2. The original series was stationary

The ADF test produced a p-value below 0.05, so non-seasonal differencing was not required for the selected model.

### 3. SARIMA outperformed SARIMAX

Although price and event variables were considered as external predictors, the tested SARIMAX models performed worse than the selected SARIMA model on the chronological validation set.

This demonstrates that additional variables do not necessarily improve out-of-sample forecasting performance.

### 4. Forecasting can directly support inventory decisions

The demand forecast was converted into:

```text
Forecast
   ↓
Lead-Time Demand
   ↓
Safety Stock
   ↓
Reorder Point
   ↓
Replenishment Quantity
```

### 5. Higher service levels require more inventory

Increasing the service-level target from 90% to 99% increases safety stock and the reorder point.

This represents the practical trade-off between inventory availability and inventory carrying requirements.

---

# Limitations

The project has several limitations:

1. Only one product-store combination was modeled in detail.
2. The M5 dataset does not provide actual inventory levels.
3. Lead time and inventory cost parameters are assumed.
4. The inventory uncertainty calculation uses an approximate residual-based approach.
5. The inventory simulation covers a 28-day evaluation period.
6. The final model is based on classical statistical time-series forecasting rather than deep learning.
7. SARIMAX external variables did not improve validation performance for the selected series.
8. The current system does not optimize replenishment policy using a full cost-based optimization algorithm.
9. The model does not explicitly model all possible promotions, price elasticity, or external economic factors.

---

# Future Improvements

Potential extensions include:

* Modeling multiple product-store combinations simultaneously.
* Incorporating richer calendar and promotion features.
* Testing price elasticity more explicitly.
* Using rolling-origin cross-validation.
* Developing probabilistic forecasting models.
* Simulating forecast uncertainty using Monte Carlo methods.
* Using dynamic inventory optimization.
* Adding supplier lead-time variability.
* Incorporating actual inventory data when available.
* Testing gradient boosting models such as XGBoost or LightGBM.
* Comparing classical forecasting models with deep learning approaches such as LSTM or Temporal Fusion Transformer.
* Developing automated product-level inventory recommendations.

---

# Conclusion

This project demonstrates an end-to-end approach to demand forecasting and inventory planning.

Rather than treating forecasting as an isolated machine learning task, the system connects statistical forecasting with a practical business decision:

```text
Historical Demand
        ↓
Time-Series Analysis
        ↓
SARIMA / SARIMAX
        ↓
Model Evaluation
        ↓
Demand Forecast
        ↓
Uncertainty Estimation
        ↓
Safety Stock
        ↓
Reorder Point
        ↓
EOQ
        ↓
Inventory Simulation
        ↓
Business Decision
```

The final SARIMA model achieved a 28-day future MAE of approximately 11.4 units and RMSE of approximately 14.1 units for the selected product-store combination.

The resulting inventory policy achieved a 100% fill rate with zero stockout units during the evaluated 28-day period under the stated assumptions.

The project demonstrates how time-series forecasting can be integrated with inventory planning to transform historical demand data into actionable replenishment decisions.

---

# Author

**Saloni Pandagade**

B.Tech — Metallurgical & Materials Engineering
Indian Institute of Technology Bhubaneswar

GitHub: `https://github.com/salonipandagale`

Portfolio: `https://portfolio-wi2l.onrender.com`

LinkedIn: `https://www.linkedin.com/in/saloni-pandagale-4010a1296/`
