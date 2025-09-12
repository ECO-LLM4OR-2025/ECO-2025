import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("terracotta_jars")

# Parameters
shaping_thin = 50
shaping_stubby = 30
baking_thin = 90
baking_stubby = 150
shaping_capacity = 3000
baking_capacity = 4000
profit_thin = 5
profit_stubby = 9

# Variables
x_thin = model.addVar(vtype=GRB.INTEGER, name="x_thin", lb=0)
x_stubby = model.addVar(vtype=GRB.INTEGER, name="x_stubby", lb=0)

# Constraints
model.addConstr(shaping_thin * x_thin + shaping_stubby * x_stubby <= shaping_capacity, name="shaping")
model.addConstr(baking_thin * x_thin + baking_stubby * x_stubby <= baking_capacity, name="baking")

# Objective
model.setObjective(profit_thin * x_thin + profit_stubby * x_stubby, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Write optimal objective value to file
with open('ref_optimal_value.txt', 'w') as f:
    if model.status == GRB.OPTIMAL:
        f.write(str(model.objVal))
    else:
        f.write("No optimal solution found")