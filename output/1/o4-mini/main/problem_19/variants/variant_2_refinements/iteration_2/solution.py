import gurobipy as gp
from gurobipy import GRB, GurobiError
import math

# Data
manufacturers    = ['A', 'B', 'C']
chairs_per_order = {'A': 15, 'B': 10, 'C': 10}
cost_per_chair   = {'A': 50, 'B': 45, 'C': 40}
min_chairs       = 100
max_chairs       = 500

# Compute a tighter big-M for each manufacturer
M = {i: math.ceil(max_chairs / chairs_per_order[i]) for i in manufacturers}

# Precompute cost per order = cost per chair × chairs per order
cost_per_order = {i: cost_per_chair[i] * chairs_per_order[i] for i in manufacturers}

try:
    # Initialize model
    model = gp.Model("chair_ordering")
    # (Optional) suppress detailed Gurobi output
    model.setParam('OutputFlag', 0)

    # Decision variables
    # x[i] = # of orders (integer)
    x = model.addVars(manufacturers, vtype=GRB.INTEGER, lb=0, name="x")
    # y[i] = 1 if we place at least one order from i, else 0
    y = model.addVars(manufacturers, vtype=GRB.BINARY, name="y")

    # Link x and y:
    #   if y[i] = 0 then x[i] = 0
    #   if y[i] = 1 then 1 ≤ x[i] ≤ M[i]
    for i in manufacturers:
        model.addConstr(x[i] >= y[i], name=f"link_min_{i}")
        model.addConstr(x[i] <= M[i] * y[i], name=f"link_max_{i}")

    # Logical implications
    # If we use A, we must use B
    model.addConstr(y['A'] <= y['B'], name="A_implies_B")
    # If we use B, we must use C
    model.addConstr(y['B'] <= y['C'], name="B_implies_C")

    # If we use A, we must order at least 10 chairs from B
    model.addConstr(
        chairs_per_order['B'] * x['B'] >= 10 * y['A'],
        name="min_B_chairs_if_A"
    )

    # Total chairs constraints
    total_chairs = gp.quicksum(chairs_per_order[i] * x[i] for i in manufacturers)
    model.addConstr(total_chairs >= min_chairs, name="min_chairs")
    model.addConstr(total_chairs <= max_chairs, name="max_chairs")

    # Objective: minimize total cost = sum(cost_per_order[i] * x[i])
    model.setObjective(
        gp.quicksum(cost_per_order[i] * x[i] for i in manufacturers),
        GRB.MINIMIZE
    )

    # Optimize
    model.optimize()

    # Extract optimal value if available
    optimal_value = None
    if model.status == GRB.OPTIMAL:
        optimal_value = model.objVal

except GurobiError:
    optimal_value = None

# Write only the numeric optimal value to file
try:
    with open('ref_optimal_value.txt', 'w') as f:
        if optimal_value is not None:
            f.write(str(optimal_value))
except IOError:
    # Silently ignore file I/O errors
    pass