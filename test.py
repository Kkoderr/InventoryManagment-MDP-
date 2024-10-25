from driver import Driver

mdp = Driver.InventoryMDP(
    max_inventory=10,
    demand_probs={0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2},
    spoilage_rate=0.05,
    order_cost=2,
    holding_cost=0.5,
    sell_price=5,
    discount=0.95
)

policy, values = mdp.value_iteration()
print("Optimal Policy:", policy)
print("Value Function:", values)