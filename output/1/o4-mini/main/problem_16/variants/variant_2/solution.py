import gurobipy as gp
from gurobipy import GRB

# Data
T = [1, 2, 3]
c = {1: 8, 2: 6, 3: 9}    # purchase prices
s = {1: 9, 2: 8, 3: 10}   # selling prices
I0 = 200                  # initial inventory
Cap = 500                 # warehouse capacity

# Model
m = gp.Model("Purchasing_and_Sales_Plan")

# Decision variables
P = m.addVars(T, name="P", lb=0, vtype=GRB.CONTINUOUS)  # purchased units
S = m.addVars(T, name="S", lb=0, vtype=GRB.CONTINUOUS)  # sold units
I = m.addVars(T, name="I", lb=0, vtype=GRB.CONTINUOUS)  # end inventory

# Constraints
# Inventory balance
for t in T:
    if t == 1:
        m.addConstr(I[t] == I0 + P[t] - S[t], name=f"inv_balance_{t}")
    else:
        m.addConstr(I[t] == I[t-1] + P[t] - S[t], name=f"inv_balance_{t}")

# Warehouse capacity
for t in T:
    m.addConstr(I[t] <= Cap, name=f"capacity_{t}")

# Objective: maximize profit
obj = gp.quicksum(s[t] * S[t] - c[t] * P[t] for t in T)
m.setObjective(obj, GRB.MAXIMIZE)

# Solve
m.optimize()

# Write optimal objective value to file
if m.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(m.objVal))
else:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("No optimal solution found.")