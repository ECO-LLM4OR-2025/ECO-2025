import gurobipy as gp
from gurobipy import GRB, GurobiError

# -----------------------------
# Data Definition
# -----------------------------
# Time periods
T = [1, 2, 3]

# Purchase prices by month (Yuan/unit)
b = {1: 8, 2: 6, 3: 9}

# Selling prices by month (Yuan/unit)
p = {1: 9, 2: 8, 3: 10}

# Forecasted demand caps (units) – assumed since original problem gives no explicit demand limit
D = {1: 180, 2: 220, 3: 210}

# Holding cost per unit held from month t to t+1 (Yuan/unit/month)
# This penalizes stockpiling and curbs inter-month arbitrage
h = {1: 1.0, 2: 1.0, 3: 0.0}

# Initial inventory at time 0
s0 = 200

# Warehouse capacity (units)
S_max = 500

# Cost of the initial stock – we assume it was acquired at the month-1 purchase price
# This is a constant offset in the objective (does not affect decision variables)
initial_inventory_cost = b[1] * s0

try:
    # -----------------------------
    # Model Initialization
    # -----------------------------
    model = gp.Model("Q1_Purchasing_Sales_Plan")
    model.Params.OutputFlag = 0   # turn off Gurobi solver output

    # -----------------------------
    # Decision Variables
    # -----------------------------
    # x[t]: units purchased in month t, integer
    x = model.addVars(T, vtype=GRB.INTEGER, lb=0, name="x")

    # y[t]: units sold in month t, integer, cannot exceed forecasted demand
    y = model.addVars(T, vtype=GRB.INTEGER, lb=0, ub=D, name="y")

    # s[t]: end-of-month inventory, continuous, cannot exceed capacity
    s = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, ub=S_max, name="s")

    # -----------------------------
    # Constraints
    # -----------------------------
    # Inventory flow balance
    # Month 1
    model.addConstr(
        s[1] == s0 + x[1] - y[1],
        name="inventory_balance_1"
    )
    # Months 2 and 3
    for t in [2, 3]:
        model.addConstr(
            s[t] == s[t-1] + x[t] - y[t],
            name=f"inventory_balance_{t}"
        )

    # Terminal inventory must be zero
    model.addConstr(s[3] == 0, name="terminal_inventory_zero")

    # -----------------------------
    # Objective Function
    # -----------------------------
    # Maximize: revenue - purchase cost - holding cost - initial inventory cost
    revenue    = gp.quicksum(p[t] * y[t]       for t in T)
    purchase   = gp.quicksum(b[t] * x[t]       for t in T)
    holding    = gp.quicksum(h[t] * s[t]       for t in T)
    constant   = initial_inventory_cost

    model.setObjective(
        revenue - purchase - holding - constant,
        GRB.MAXIMIZE
    )

    # -----------------------------
    # Optimize
    # -----------------------------
    model.optimize()

    # -----------------------------
    # Output Optimal Value
    # -----------------------------
    if model.Status == GRB.OPTIMAL:
        with open('ref_optimal_value.txt', 'w') as fout:
            # Write only the numeric optimal objective
            fout.write(f"{model.ObjVal:.6f}")
    else:
        raise GurobiError(f"Optimization ended with status {model.Status}")

except GurobiError as e:
    print(f"Gurobi error: {e}")
except Exception as e:
    print(f"General exception: {e}")