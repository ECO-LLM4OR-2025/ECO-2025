import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("resource_allocation")

# Decision variables
x_X = model.addVar(vtype=GRB.INTEGER, name="x_X", lb=0, ub=700)
x_Y = model.addVar(vtype=GRB.INTEGER, name="x_Y", lb=0, ub=500)

# Budget constraint
model.addConstr(x_X + x_Y <= 1000, name="budget")

# Minimum excess requirement
model.addConstr(x_X - x_Y >= 200, name="excess")

# Objective: minimize total cost
model.setObjective(50 * x_X + 30 * x_Y, GRB.MINIMIZE)

# Optimize model
model.optimize()

# Write optimal value to file
with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(int(model.objVal)))