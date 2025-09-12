import gurobipy as gp
from gurobipy import GRB

# Data
profits = {'chair': 43, 'dresser': 52}
stain_req = {'chair': 1.4, 'dresser': 1.1}
wood_req = {'chair': 2,   'dresser': 3}
stain_avail = 17
wood_avail = 11

# Model
model = gp.Model()

# Decision variables
x = model.addVars(profits.keys(), lb=0, vtype=GRB.CONTINUOUS, name='x')

# Constraints
model.addConstr(
    gp.quicksum(stain_req[p] * x[p] for p in profits) <= stain_avail,
    name='StainCapacity'
)
model.addConstr(
    gp.quicksum(wood_req[p] * x[p] for p in profits) <= wood_avail,
    name='WoodCapacity'
)

# Objective
model.setObjective(
    gp.quicksum(profits[p] * x[p] for p in profits),
    GRB.MAXIMIZE
)

# Solve
model.optimize()

# Save optimal value
opt_val = model.objVal
with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(opt_val))