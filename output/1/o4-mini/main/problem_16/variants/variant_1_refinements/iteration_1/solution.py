import gurobipy as gp
from gurobipy import GRB, GurobiError

# Data
T = [1, 2, 3]  # months
p_buy  = {1: 8, 2: 6, 3: 9}   # purchase prices
p_sell = {1: 9, 2: 8, 3: 10}  # selling prices
I0      = 200                 # initial inventory
C       = 500                 # warehouse capacity
# Monthly demand limits (units) – these cap how much can actually be sold each month
D       = {1: 200, 2: 250, 3: 300}

try:
    # Create model
    m = gp.Model("quarterly_inventory")
    m.ModelSense = GRB.MAXIMIZE
    m.setParam('OutputFlag', 0)  # silent solve

    # Decision variables
    q = m.addVars(T, name="q", lb=0.0)  # quantities purchased
    s = m.addVars(T, name="s", lb=0.0)  # quantities sold
    I = m.addVars(T, name="I", lb=0.0)  # end-of-month inventory

    # Inventory balance constraints
    for t in T:
        if t == 1:
            m.addConstr(I0 + q[t] - s[t] == I[t], name=f"balance_{t}")
        else:
            m.addConstr(I[t-1] + q[t] - s[t] == I[t], name=f"balance_{t}")

    # Warehouse capacity constraints
    m.addConstrs((I[t] <= C for t in T), name="capacity")

    # Demand constraints to cap sales and bound the model
    m.addConstrs((s[t] <= D[t] for t in T), name="demand_limit")

    # Objective: maximize total profit over three months
    profit = gp.quicksum(p_sell[t] * s[t] - p_buy[t] * q[t] for t in T)
    m.setObjective(profit)

    # Solve
    m.optimize()

    # Write out the optimal value (numeric only) or blank if no optimum
    with open('ref_optimal_value.txt', 'w') as f:
        if m.status == GRB.OPTIMAL:
            f.write(str(m.objVal))
        else:
            f.write("")

except GurobiError as e:
    # If Gurobi fails, write nothing
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("") 
    print("Gurobi error:", e)
except Exception as ex:
    # Catch-all for any other errors
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("") 
    print("Unexpected error:", ex)