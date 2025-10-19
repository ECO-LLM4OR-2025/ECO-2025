import gurobipy as gp
from gurobipy import GRB

# Data
T = [1, 2, 3]
p_buy = {1: 8, 2: 6, 3: 9}
p_sell = {1: 9, 2: 8, 3: 10}
I0 = 200
C = 500

# Model
m = gp.Model("quarterly_inventory")

# Decision variables
q = m.addVars(T, lb=0, name="q")    # purchase quantities
s = m.addVars(T, lb=0, name="s")    # sales quantities
I = m.addVars(T, lb=0, name="I")    # end-of-month inventories

# Inventory balance constraints
for t in T:
    if t == 1:
        m.addConstr(I0 + q[t] - s[t] == I[t], name=f"balance_{t}")
    else:
        m.addConstr(I[t-1] + q[t] - s[t] == I[t], name=f"balance_{t}")

# Capacity constraints
for t in T:
    m.addConstr(I[t] <= C, name=f"cap_{t}")

# Objective: maximize profit
m.setObjective(gp.quicksum(p_sell[t] * s[t] - p_buy[t] * q[t] for t in T), GRB.MAXIMIZE)

# Solve
m.optimize()

# Save optimal value
if m.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(m.objVal))
else:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("No optimal solution found.")