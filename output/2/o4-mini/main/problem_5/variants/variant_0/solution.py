import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("flooring_profit_maximization")

# Decision variables
x_h = model.addVar(name="hardwood_sqft", lb=0.0, vtype=GRB.CONTINUOUS)
x_v = model.addVar(name="vinyl_sqft", lb=0.0, vtype=GRB.CONTINUOUS)

# Parameters
profit_h = 2.5
profit_v = 3.0
demand_h_min = 20000
demand_v_min = 10000
ship_total_min = 60000
prod_h_max = 50000
prod_v_max = 30000

# Constraints
model.addConstr(x_h >= demand_h_min, name="hardwood_min_demand")
model.addConstr(x_v >= demand_v_min, name="vinyl_min_demand")
model.addConstr(x_h + x_v >= ship_total_min, name="shipping_contract")
model.addConstr(x_h <= prod_h_max, name="hardwood_capacity")
model.addConstr(x_v <= prod_v_max, name="vinyl_capacity")

# Objective: maximize profit
model.setObjective(profit_h * x_h + profit_v * x_v, GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Save the optimal value to file
if model.status == GRB.OPTIMAL:
    with open("ref_optimal_value.txt", "w") as f:
        f.write(str(model.objVal))
else:
    with open("ref_optimal_value.txt", "w") as f:
        f.write("No optimal solution found")