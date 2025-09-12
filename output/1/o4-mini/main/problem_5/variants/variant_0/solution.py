import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("flooring_production")

# Decision variables
x_h = model.addVar(lb=0, name="hardwood_sqft")
x_v = model.addVar(lb=0, name="vinyl_sqft")

# Constraints
model.addConstr(x_h >= 20000, name="hardwood_demand")
model.addConstr(x_v >= 10000, name="vinyl_demand")
model.addConstr(x_h + x_v >= 60000, name="total_shipping")
model.addConstr(x_h <= 50000, name="hardwood_capacity")
model.addConstr(x_v <= 30000, name="vinyl_capacity")

# Objective: maximize profit
model.setObjective(2.5 * x_h + 3.0 * x_v, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Retrieve and save optimal value
if model.status == GRB.OPTIMAL:
    opt_val = model.objVal
else:
    opt_val = None

with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(opt_val))