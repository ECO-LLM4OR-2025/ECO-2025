import gurobipy as gp
from gurobipy import GRB

# Create model
m = gp.Model("terracotta_jars")

# Decision variables
x_thin   = m.addVar(vtype=GRB.INTEGER, name="thin",    lb=0)
x_stubby = m.addVar(vtype=GRB.INTEGER, name="stubby",  lb=0)

# Parameters
shaping_time_thin   = 50
shaping_time_stubby = 30
baking_time_thin    = 90
baking_time_stubby  = 150
avail_shaping       = 3000
avail_baking        = 4000
profit_thin         = 5
profit_stubby       = 9

# Constraints
m.addConstr(shaping_time_thin * x_thin 
            + shaping_time_stubby * x_stubby 
            <= avail_shaping, name="shaping_capacity")

m.addConstr(baking_time_thin * x_thin 
            + baking_time_stubby * x_stubby 
            <= avail_baking, name="baking_capacity")

# Objective: maximize profit
m.setObjective(profit_thin * x_thin 
               + profit_stubby * x_stubby, GRB.MAXIMIZE)

# Solve
m.optimize()

# Save optimal value
if m.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(m.objVal))