import gurobipy as gp
from gurobipy import GRB, GurobiError
import math

# Data definition
manufacturers    = ['A', 'B', 'C']
chairs_per_order = {'A': 15, 'B': 10, 'C': 10}
cost_per_chair   = {'A': 50, 'B': 45, 'C': 40}
min_chairs       = 100
max_chairs       = 500

# Precompute big-M per manufacturer (max possible orders)
M = {m: math.ceil(max_chairs / chairs_per_order[m]) for m in manufacturers}

# Precompute cost per order
cost_per_order = {m: cost_per_chair[m] * chairs_per_order[m] for m in manufacturers}

optimal_value = None

try:
    # Initialize Gurobi model
    model = gp.Model("chair_ordering")
    # Suppress solver output for clarity
    model.setParam('OutputFlag', 0)

    # Decision variables:
    # x[m] = integer number of orders from manufacturer m
    # y[m] = binary flag, 1 if any order is placed from m, 0 otherwise
    x = model.addVars(manufacturers, vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addVars(manufacturers, vtype=GRB.BINARY, name="y")

    # Link x and y with tight big-M:
    #   y[m] = 0 ⇒ x[m] = 0
    #   y[m] = 1 ⇒ 1 ≤ x[m] ≤ M[m]
    for m in manufacturers:
        model.addConstr(x[m] >= y[m],           name=f"link_min_{m}")
        model.addConstr(x[m] <= M[m] * y[m],    name=f"link_max_{m}")

    # Logical implications between manufacturers:
    # If we order from A, we must order from B; if from B, must order from C
    model.addConstr(y['A'] <= y['B'], name="A_implies_B")
    model.addConstr(y['B'] <= y['C'], name="B_implies_C")

    # Total chairs constraints
    total_chairs = gp.quicksum(chairs_per_order[m] * x[m] for m in manufacturers)
    model.addConstr(total_chairs >= min_chairs, name="min_chairs")
    model.addConstr(total_chairs <= max_chairs, name="max_chairs")

    # Objective: minimize total cost
    model.setObjective(
        gp.quicksum(cost_per_order[m] * x[m] for m in manufacturers),
        GRB.MINIMIZE
    )

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        optimal_value = model.objVal

except GurobiError as e:
    # Model could not be optimized
    optimal_value = None

# Write only the numeric optimal value to file
try:
    with open('ref_optimal_value.txt', 'w') as f:
        if optimal_value is not None:
            f.write(str(optimal_value))
except IOError:
    pass