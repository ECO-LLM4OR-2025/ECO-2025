import gurobipy as gp
from gurobipy import GRB

# Data
T = [1, 2, 3]
b = {1: 8, 2: 6, 3: 9}    # purchase prices
p = {1: 9, 2: 8, 3: 10}   # selling prices
s0 = 200                 # initial inventory
S_max = 500              # warehouse capacity

# Model
model = gp.Model("Purchasing_Sales_Plan")

# Decision variables
x = model.addVars(T, lb=0, name="purchase")   # x[t]: units purchased in month t
y = model.addVars(T, lb=0, name="sales")      # y[t]: units sold in month t
s = model.addVars(T, lb=0, name="inventory")  # s[t]: inventory at end of month t

# Inventory balance constraints
for t in T:
    if t == 1:
        model.addConstr(s[t] == s0 + x[t] - y[t], name=f"inv_balance_{t}")
    else:
        model.addConstr(s[t] == s[t-1] + x[t] - y[t], name=f"inv_balance_{t}")

# Capacity constraints
for t in T:
    model.addConstr(s[t] <= S_max, name=f"capacity_{t}")

# Objective: maximize profit
model.setObjective(gp.quicksum(p[t] * y[t] - b[t] * x[t] for t in T), GRB.MAXIMIZE)

# Optimize
model.optimize()

# Write optimal value to file
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))