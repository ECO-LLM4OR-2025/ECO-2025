import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("flooring_production")

# Decision variables
x_h = model.addVar(lb=0, ub=50000, name="hardwood")  # Hardwood sq ft
x_v = model.addVar(lb=0, ub=30000, name="vinyl")     # Vinyl sq ft

# Constraints
model.addConstr(x_h >= 20000, name="min_hardwood")       # Min hardwood demand
model.addConstr(x_v >= 10000, name="min_vinyl")          # Min vinyl demand
model.addConstr(x_h + x_v >= 60000, name="min_total")    # Shipping requirement

# Objective: maximize profit
model.setObjective(2.5 * x_h + 3.0 * x_v, GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Write optimal value to file
if model.status == GRB.OPTIMAL:
    with open("ref_optimal_value.txt", "w") as f:
        f.write(str(model.objVal))