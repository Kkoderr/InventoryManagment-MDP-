import numpy as np

class MultiItemInventoryMDP:
    def __init__(self, items):
        self.items = items 

    def states(self, max_inventory):
        return range(max_inventory + 1)

    def actions(self, state, max_inventory):
        return range(max_inventory - state + 1)

    def succProbReward(self, state, action, item_params):
        result = []
        new_stock = min(state + action, item_params['max_inventory'])
        cost = action * item_params['order_cost'] - new_stock * item_params['holding_cost']
        
        for demand, prob in item_params['demand_probs'].items():
            sold = min(demand, new_stock)
            revenue = sold * item_params['sell_price']
            new_state = max(new_stock - sold, 0)
            result.append((new_state, prob, revenue + cost))
        
        return result

    def value_iteration(self, item_params):
        max_inventory = item_params['max_inventory']
        V = {s: 0 for s in self.states(max_inventory)}

        def Q(state, action):
            return sum(prob * (reward + item_params['discount'] * V[newState]) 
                       for newState, prob, reward in self.succProbReward(state, action, item_params))

        while True:
            newV = {state: max(Q(state, action) for action in self.actions(state, max_inventory)) for state in self.states(max_inventory)}
            if max(abs(V[state] - newV[state]) for state in self.states(max_inventory)) < 1e-5:
                break
            V = newV
        
        policy = {state: max((Q(state, action), action) for action in self.actions(state, max_inventory))[1] for state in self.states(max_inventory)}
        return policy, V