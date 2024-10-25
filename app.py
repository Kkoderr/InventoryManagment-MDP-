import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from driver import Driver

st.title("Smart Inventory Management MDP")

# Number of items
num_items = st.sidebar.number_input("Number of Items", min_value=1, max_value=10, value=2)

# Define parameters for each item
items = {}
for i in range(num_items):
    st.sidebar.header(f"Item {i+1} Parameters")
    max_inventory = st.sidebar.slider(f"Max Inventory (Item {i+1})", min_value=5, max_value=50, value=10, step=1)
    spoilage_rate = st.sidebar.slider(f"Spoilage Rate (Item {i+1})", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    order_cost = st.sidebar.number_input(f"Order Cost per unit (Item {i+1})", value=2.0)
    holding_cost = st.sidebar.number_input(f"Holding Cost per unit (Item {i+1})", value=0.5)
    sell_price = st.sidebar.number_input(f"Selling Price per unit (Item {i+1})", value=5.0)
    discount_factor = st.sidebar.slider(f"Discount Factor (Item {i+1})", min_value=0.8, max_value=1.0, value=0.95)

    # Demand Probabilities
    st.sidebar.subheader(f"Demand Probabilities (Item {i+1})")
    demand_prob_low = st.sidebar.slider(f"Low Demand (0-1 units) (Item {i+1})", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    demand_prob_med = st.sidebar.slider(f"Medium Demand (2-3 units) (Item {i+1})", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    demand_prob_high = st.sidebar.slider(f"High Demand (4-5 units) (Item {i+1})", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    total_prob = demand_prob_low + demand_prob_med + demand_prob_high
    demand_probs = {
        0: demand_prob_low / total_prob * 0.5, 1: demand_prob_low / total_prob * 0.5,
        2: demand_prob_med / total_prob * 0.5, 3: demand_prob_med / total_prob * 0.5,
        4: demand_prob_high / total_prob * 0.5, 5: demand_prob_high / total_prob * 0.5
    }

    items[f"Item_{i+1}"] = {
        'max_inventory': max_inventory,
        'spoilage_rate': spoilage_rate,
        'order_cost': order_cost,
        'holding_cost': holding_cost,
        'sell_price': sell_price,
        'discount': discount_factor,
        'demand_probs': demand_probs
    }

# Instantiate and run MDP model for each item
mdp = Driver.MultiItemInventoryMDP(items)
results = {item_name: mdp.value_iteration(params) for item_name, params in items.items()}

# Display each item’s optimal policy and value function in a table
for item_name, (policy, values) in results.items():
    with st.expander(f"{item_name} "):
        # Combine policy and value into a DataFrame
        df = pd.DataFrame({
            "State": list(values.keys()),
            "Value": list(values.values()),
            "Policy (Order Quantity)": [policy[state] for state in values.keys()]
        })
        st.table(df)