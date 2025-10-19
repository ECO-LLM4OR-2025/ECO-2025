import gurobipy as gp
from gurobipy import GRB

try:
    # Data
    T = [1, 2, 3]                       # Planning horizon: months 1–3
    b = {1: 8,  2: 6,  3: 9}            # Purchase prices
    p = {1: 9,  2: 8,  3: 10}           # Selling prices
    D = {1: 180, 2: 220, 3: 210}        # Forecasted demand limits
    s0 = 200                            # Initial inventory
    S_max = 500                         # Warehouse capacity
    Q_max = 300                         # Max units we can purchase per month

    # Create model
    model = gp.Model("Purchasing_Sales_Plan")

    # Decision variables
    x = model.addVars(T, lb=0, ub=Q_max, name="purchase")   # Units purchased
    y = model.addVars(T, lb=0, ub=gp.quicksum(D.values()), name="sales")  # Units sold
    s = model.addVars(T, lb=0, ub=S_max, name="inventory")   # End-of-month inventory

    # Inventory balance constraints
    for t in T:
        if t == 1:
            model.addConstr(s[t] == s0 + x[t] - y[t],
                            name="inv_balance_1")
        else:
            model.addConstr(s[t] == s[t-1] + x[t] - y[t],
                            name=f"inv_balance_{t}")

    # Capacity constraints (already in var ub, but keep for clarity)
    for t in T:
        model.addConstr(s[t] <= S_max, name=f"storage_cap_{t}")

    # Demand constraints to cap sales
    for t in T:
        model.addConstr(y[t] <= D[t], name=f"demand_cap_{t}")

    # Terminal inventory requirement: no leftover stock at end of month 3
    model.addConstr(s[3] == 0, name="terminal_inventory_zero")

    # Objective: maximize total profit
    profit_expr = gp.quicksum(p[t] * y[t] - b[t] * x[t] for t in T)
    model.setObjective(profit_expr, GRB.MAXIMIZE)

    # Solve
    model.setParam('OutputFlag', 0)  # silence Gurobi output
    model.optimize()

    # Check for optimal solution and write to file
    if model.Status == GRB.OPTIMAL:
        with open('ref_optimal_value.txt', 'w') as f:
            f.write(f"{model.ObjVal}")
    else:
        raise gp.GurobiError("No optimal solution found.")

except gp.GurobiError as e:
    print(f"Gurobi exception: {e}")
except Exception as ex:
    print(f"General exception: {ex}")