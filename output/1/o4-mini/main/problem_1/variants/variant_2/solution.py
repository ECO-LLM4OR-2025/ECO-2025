import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model()

# Decision variables
x = model.addVar(vtype=GRB.INTEGER, name="x_X", lb=0, ub=700)
y = model.addVar(vtype=GRB.INTEGER, name="x_Y", lb=0, ub=500)

# Constraints
model.addConstr(x + y <= 1000, name="total_resource")
model.addConstr(x - y >= 200, name="excess_requirement")

# Objective: minimize cost
model.setObjective(50 * x + 30 * y, GRB.MINIMIZE)

# Optimize
model.optimize()

# Write optimal value to file
with open('ref_optimal_value.txt', 'w') as f:
    f.write(f"{model.objVal:.0f}")