import gurobipy as gp
from gurobipy import GRB

# Create Gurobi model
model = gp.Model()

# Decision variables
x_thin = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_thin")
x_stubby = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_stubby")

# Constraints
model.addConstr(50 * x_thin + 30 * x_stubby <= 3000, name="shaping_capacity")
model.addConstr(90 * x_thin + 150 * x_stubby <= 4000, name="baking_capacity")

# Objective: maximize profit
model.setObjective(5 * x_thin + 9 * x_stubby, GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Write the optimal objective value to file
with open("ref_optimal_value.txt", "w") as f:
    f.write(str(model.objVal))