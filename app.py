import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

# 2. LOAD SAVED ARTIFACTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(MODEL_DIR, "final_sarima_model.pkl")
    )


@st.cache_data
def load_historical_data():
    return joblib.load(
        os.path.join(MODEL_DIR, "historical_demand.pkl")
    )


@st.cache_data
def load_inventory_parameters():
    return joblib.load(
        os.path.join(MODEL_DIR, "inventory_parameters.pkl")
    )


@st.cache_data
def load_model_comparison():
    return joblib.load(
        os.path.join(MODEL_DIR, "model_comparison.pkl")
    )


@st.cache_data
def load_future_results():
    return joblib.load(
        os.path.join(MODEL_DIR, "future_forecast_results.pkl")
    )

# Load everything
model = load_model()
historical_data = load_historical_data()
inventory_params = load_inventory_parameters()
model_comparison = load_model_comparison()
future_results = load_future_results()

# 3. EXTRACT PARAMETERS

selected_item = inventory_params["selected_item"]
selected_store = inventory_params["selected_store"]

average_daily_demand = inventory_params["average_daily_demand"]
annual_demand = inventory_params["annual_demand"]

lead_time = inventory_params["lead_time"]
residual_std = inventory_params["residual_std"]

ordering_cost = inventory_params["ordering_cost"]
holding_cost = inventory_params["holding_cost"]

base_eoq = inventory_params["eoq"]

# 4. SIDEBAR

st.sidebar.title("📦 Inventory Controls")

st.sidebar.markdown("### Product")

st.sidebar.info(
    f"**Item:** {selected_item}\n\n"
    f"**Store:** {selected_store}"
)

st.sidebar.markdown("### Inventory Parameters")

service_level = st.sidebar.selectbox(
    "Target Service Level",
    [0.90, 0.95, 0.99],
    index=1,
    format_func=lambda x: f"{x * 100:.0f}%"
)

lead_time_input = st.sidebar.number_input(
    "Lead Time (days)",
    min_value=1,
    max_value=30,
    value=int(lead_time),
    step=1
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=28,
    value=28
)


# 5. CALCULATE INVENTORY POLICY

# Z-values for common service levels
z_values = {
    0.90: 1.281552,
    0.95: 1.644854,
    0.99: 2.326348
}

z_value = z_values[service_level]


# Use the forecast for the selected lead time
forecast_values = future_results["forecast"].values

lead_time_forecast = forecast_values[:lead_time_input]

lead_time_demand = np.sum(lead_time_forecast)

lead_time_std = residual_std * np.sqrt(lead_time_input)

safety_stock = z_value * lead_time_std

reorder_point = lead_time_demand + safety_stock

eoq = base_eoq


# 6. HEADER

st.title("📦 Demand Forecasting & Inventory Optimization")

st.markdown(
    """
    **SARIMA-based demand forecasting and inventory planning system**
    
    This dashboard forecasts product demand and converts the forecast
    into an inventory replenishment policy using safety stock,
    reorder point and Economic Order Quantity (EOQ).
    """
)

st.divider()


# 7. PROJECT OVERVIEW

st.header("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Product",
        selected_item
    )

with col2:
    st.metric(
        "Store",
        selected_store
    )

with col3:
    st.metric(
        "Historical Days",
        f"{len(historical_data):,}"
    )

with col4:
    st.metric(
        "Average Daily Demand",
        f"{average_daily_demand:.1f}"
    )


# 8. DEMAND FORECAST

st.header("📈 Demand Forecast")

st.markdown(
    f"""
    **Final Model:** SARIMA(2,0,1)(1,0,1,7)
    
    The model captures both short-term temporal dependence
    and weekly seasonal demand patterns.
    """
)


# Select requested forecast horizon
forecast_display = future_results.iloc[:forecast_horizon].copy()


fig, ax = plt.subplots(figsize=(14, 6))

# Last 90 historical observations
historical_plot = historical_data.iloc[-90:]

ax.plot(
    historical_plot.index,
    historical_plot["Demand"],
    label="Historical Demand"
)

# Forecast
ax.plot(
    forecast_display.index,
    forecast_display["forecast"],
    label="Forecast"
)

# Prediction interval
ax.fill_between(
    forecast_display.index,
    forecast_display["lower_95"],
    forecast_display["upper_95"],
    alpha=0.20,
    label="95% Prediction Interval"
)

# Actual future demand if available
if "actual_demand" in forecast_display.columns:
    ax.plot(
        forecast_display.index,
        forecast_display["actual_demand"],
        label="Actual Future Demand"
    )

ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.set_title("Demand Forecast with 95% Prediction Interval")
ax.legend()
ax.grid(alpha=0.2)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# 9. FORECAST TABLE

st.subheader("Forecast Details")

display_forecast = future_results.iloc[:forecast_horizon].copy()


if "date" in display_forecast.columns:
    display_forecast["date"] = pd.to_datetime(
        display_forecast["date"]
    )
    display_forecast = display_forecast.set_index("date")

display_forecast = display_forecast.reset_index()


if "index" in display_forecast.columns:
    display_forecast = display_forecast.rename(
        columns={"index": "Date"}
    )

elif "date" in display_forecast.columns:
    display_forecast = display_forecast.rename(
        columns={"date": "Date"}
    )

columns_to_show = [
    col for col in [
        "Date",
        "forecast",
        "lower_95",
        "upper_95",
        "actual_demand"
    ]
    if col in display_forecast.columns
]

display_forecast = display_forecast[columns_to_show]

display_forecast = display_forecast.rename(
    columns={
        "forecast": "Forecast",
        "lower_95": "Lower 95%",
        "upper_95": "Upper 95%",
        "actual_demand": "Actual Demand"
    }
)

display_forecast = display_forecast.loc[
    :,
    ~display_forecast.columns.duplicated()
]

st.dataframe(
    display_forecast.round(2),
    use_container_width=True,
    hide_index=True
)

# 10. FORECAST PERFORMANCE

st.header("🎯 Forecast Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "28-Day MAE",
        "11.40"
    )

with col2:
    st.metric(
        "28-Day RMSE",
        "14.08"
    )

with col3:
    st.metric(
        "28-Day MAPE",
        "15.54%"
    )


st.markdown(
    """
    **Interpretation:** On the 28-day future evaluation period,
    the model's average absolute forecasting error was approximately
    11.4 units per day.
    """
)


# 11. SARIMA VS SARIMAX

st.subheader("SARIMA vs SARIMAX")

st.dataframe(
    model_comparison.round(3),
    use_container_width=True,
    hide_index=True
)


# Highlight best model
best_model = model_comparison.loc[
    model_comparison["Validation_MAE"].idxmin(),
    "Model"
]

st.success(
    f"Based on validation MAE, the selected model is: **{best_model}**"
)


# 12. INVENTORY OPTIMIZATION

st.header("📦 Inventory Optimization")

st.markdown(
    """
    The forecast is converted into an inventory replenishment policy.
    The policy uses lead-time demand, safety stock and EOQ.
    """
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Lead-Time Demand",
        f"{lead_time_demand:.0f} units"
    )

with col2:
    st.metric(
        "Safety Stock",
        f"{safety_stock:.0f} units"
    )

with col3:
    st.metric(
        "Reorder Point",
        f"{reorder_point:.0f} units"
    )

with col4:
    st.metric(
        "EOQ",
        f"{eoq:.0f} units"
    )


st.info(
    f"""
    **Inventory Policy**
    
    When inventory position falls to approximately
    **{reorder_point:.0f} units**, place an order of approximately
    **{eoq:.0f} units**.
    
    Target service level: **{service_level * 100:.0f}%**
    
    Lead time: **{lead_time_input} days**
    """
)


# 13. INVENTORY POLICY EXPLANATION

st.subheader("How the Inventory Policy Works")

policy_col1, policy_col2, policy_col3 = st.columns(3)

with policy_col1:
    st.markdown("### 1️⃣ Lead-Time Demand")
    st.write(
        f"Expected demand during the {lead_time_input}-day "
        f"lead time is approximately **{lead_time_demand:.0f} units**."
    )

with policy_col2:
    st.markdown("### 2️⃣ Safety Stock")
    st.write(
        f"Safety stock for the selected "
        f"{service_level * 100:.0f}% service level is approximately "
        f"**{safety_stock:.0f} units**."
    )

with policy_col3:
    st.markdown("### 3️⃣ Reorder Point")
    st.write(
        f"Replenishment is triggered when inventory reaches "
        f"approximately **{reorder_point:.0f} units**."
    )


# 14. INVENTORY SIMULATION

st.header("📊 Inventory Simulation")

st.markdown(
    """
    The following simulation uses the 28-day future demand period
    to evaluate the replenishment policy.
    """
)


# Use actual future demand if available
if "actual_demand" in future_results.columns:

    simulation_data = future_results.iloc[:28].copy()

    inventory = reorder_point + eoq

    inventory_records = []

    total_stockout = 0
    total_orders = 0

    for date, row in simulation_data.iterrows():

        demand = max(float(row["actual_demand"]), 0)

        beginning_inventory = inventory

        # Demand fulfilled
        units_sold = min(beginning_inventory, demand)

        stockout_units = max(
            demand - beginning_inventory,
            0
        )

        ending_inventory = beginning_inventory - units_sold

        order_placed = 0

        # Reorder policy
        if ending_inventory <= reorder_point:

            order_placed = eoq
            ending_inventory += order_placed
            total_orders += 1

        total_stockout += stockout_units

        inventory = ending_inventory

        inventory_records.append({
            "date": date,
            "beginning_inventory": beginning_inventory,
            "demand": demand,
            "units_sold": units_sold,
            "stockout_units": stockout_units,
            "ending_inventory": ending_inventory,
            "order_placed": order_placed
        })


    simulation_df = pd.DataFrame(inventory_records)

    total_demand = simulation_df["demand"].sum()
    total_units_sold = simulation_df["units_sold"].sum()

    fill_rate = (
        total_units_sold / total_demand
        if total_demand > 0
        else 0
    )

    average_inventory = simulation_df[
        "ending_inventory"
    ].mean()

    holding_cost_total = (
        average_inventory *
        holding_cost *
        28 / 365
    )

    stockout_cost_per_unit = 5

    stockout_cost_total = (
        total_stockout *
        stockout_cost_per_unit
    )

    ordering_cost_total = (
        total_orders *
        ordering_cost
    )

    total_inventory_cost = (
        holding_cost_total +
        stockout_cost_total +
        ordering_cost_total
    )


    # --------------------------------------------------------
    # Inventory metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fill Rate",
            f"{fill_rate * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Stockout Units",
            f"{total_stockout:.0f}"
        )

    with col3:
        st.metric(
            "Orders Placed",
            f"{total_orders}"
        )

    with col4:
        st.metric(
            "Total Inventory Cost",
            f"{total_inventory_cost:.2f}"
        )


    # --------------------------------------------------------
    # Inventory chart
    # --------------------------------------------------------

    st.subheader("Inventory Level and Reorder Policy")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        simulation_df["date"],
        simulation_df["ending_inventory"],
        marker="o",
        label="Ending Inventory"
    )

    ax.axhline(
        reorder_point,
        linestyle="--",
        label="Reorder Point"
    )

    reorder_days = simulation_df[
        simulation_df["order_placed"] > 0
    ]

    if len(reorder_days) > 0:

        ax.scatter(
            reorder_days["date"],
            reorder_days["ending_inventory"],
            s=80,
            label="Reorder"
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Inventory Units")
    ax.set_title("Inventory Level and Reorder Policy")

    ax.legend()
    ax.grid(alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # Simulation table
    # --------------------------------------------------------

    st.subheader("Inventory Simulation Details")

    simulation_display = simulation_df.copy()

    simulation_display.columns = [
        col.replace("_", " ").title()
        for col in simulation_display.columns
    ]

    st.dataframe(
        simulation_display.round(2),
        use_container_width=True,
        hide_index=True
    )


# 15. SERVICE LEVEL SCENARIO ANALYSIS

st.header("📊 Service-Level Scenario Analysis")

st.markdown(
    """
    This section shows how changing the target service level
    affects safety stock and the reorder point.
    """
)


scenario_levels = [0.90, 0.95, 0.99]

scenario_results = []

for level in scenario_levels:

    z = z_values[level]

    ss = z * residual_std * np.sqrt(lead_time)

    rop = lead_time_demand + ss

    scenario_results.append({
        "Service Level": f"{level * 100:.0f}%",
        "Z Value": z,
        "Safety Stock": ss,
        "Reorder Point": rop,
        "EOQ": eoq
    })


scenario_df = pd.DataFrame(scenario_results)

st.dataframe(
    scenario_df.round(2),
    use_container_width=True,
    hide_index=True
)


# 16. SERVICE LEVEL VS REORDER POINT

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    scenario_df["Service Level"],
    scenario_df["Reorder Point"],
    marker="o"
)

ax.set_xlabel("Service Level")
ax.set_ylabel("Reorder Point")
ax.set_title("Service Level vs Reorder Point")

ax.grid(alpha=0.2)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# 17. SERVICE LEVEL VS SAFETY STOCK

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    scenario_df["Service Level"],
    scenario_df["Safety Stock"],
    marker="o"
)

ax.set_xlabel("Service Level")
ax.set_ylabel("Safety Stock")
ax.set_title("Service Level vs Safety Stock")

ax.grid(alpha=0.2)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# 18. ASSUMPTIONS

st.header("⚙️ Project Assumptions")

st.markdown(
    f"""
    - **Selected product:** {selected_item}
    - **Selected store:** {selected_store}
    - **Forecasting model:** SARIMA(2,0,1)(1,0,1,7)
    - **Forecast seasonality:** 7 days
    - **Lead time:** {lead_time} days
    - **Default service level:** 95%
    - **Ordering cost:** ${ordering_cost:.2f} per order
    - **Holding cost:** ${holding_cost:.2f} per unit per year
    - **Inventory parameters:** Business assumptions because the M5 dataset
      does not provide actual inventory, lead-time or ordering-cost data.
    """
)


# 19. FOOTER

st.divider()

st.caption(
    "Demand Forecasting & Inventory Optimization | "
    "SARIMA-based forecasting with inventory planning"
)