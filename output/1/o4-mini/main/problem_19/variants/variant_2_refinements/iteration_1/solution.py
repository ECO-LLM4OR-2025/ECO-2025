import gurobipy as gp
from gurobipy import GRB, GurobiError

# Data
manufacturers   = ['A', 'B', 'C']
chairs_per_order = {'A': 15, 'B': 10, 'C': 10}
cost_per_chair   = {'A': 50, 'B': 45, 'C': 40}
min_chairs       = 100
max_chairs       = 500

# Dynamically compute big-M: maximum possible orders for each manufacturer
M = {i: max_chairs // chairs_per_order[i] for i in manufacturers}

try:
    # Initialize model
    model = gp.Model("chair_ordering")

    # Decision variables
    # x[i]: integer # of orders from manufacturer i
    x = model.addVars(manufacturers, vtype=GRB.INTEGER, lb=0, name="orders")
    # y[i]: binary flag, 1 if we place ≥1 order from i
    y = model.addVars(manufacturers, vtype=GRB.BINARY, name="used")

    # Link x and y: if y[i]=0 then x[i]=0; if y[i]=1 then x[i]≥1 and ≤M[i]
    for i in manufacturers:
        model.addConstr(x[i] >= y[i],             name=f"min_order_{i}")
        model.addConstr(x[i] <= M[i] * y[i],       name=f"max_order_{i}")

    # Logical implications
    # If A is used then B must be used
    model.addConstr(y['A'] <= y['B'], name="A_implies_B")
    # If B is used then C must be used
    model.addConstr(y['B'] <= y['C'], name="B_implies_C")

    # Total chairs constraints
    model.addConstr(
        gp.quicksum(chairs_per_order[i] * x[i] for i in manufacturers) >= min_chairs,
        name="min_chairs"
    )
    model.addConstr(
        gp.quicksum(chairs_per_order[i] * x[i] for i in manufacturers) <= max_chairs,
        name="max_chairs"
    )

    # Objective: minimize total cost
    #   cost_per_chair × chairs_per_order × #orders
    model.setObjective(
        gp.quicksum(cost_per_chair[i] * chairs_per_order[i] * x[i]
                    for i in manufacturers),
        GRB.MINIMIZE
    )

    # Solve
    model.optimize()

    # Extract optimal cost if found
    if model.status == GRB.OPTIMAL:
        optimal_value = model.objVal
    else:
        optimal_value = None

except GurobiError:
    optimal_value = None

# Write only the numeric optimal value (no extra text) to file
try:
    with open('ref_optimal_value.txt', 'w') as f:
        if optimal_value is not None:
            f.write(str(optimal_value))
        else:
            # leave file blank if no optimal solution
            pass
except IOError:
    # silently ignore file I/O errors
    pass