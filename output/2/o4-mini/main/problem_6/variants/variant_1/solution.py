import gurobipy as gp
from gurobipy import GRB

# Model creation
model = gp.Model("meal_delivery")

# Parameters
charge_bike = 3
charge_scooter = 2
cap_bike = 8
cap_scooter = 5
total_charge = 200
min_scooters = 20

# Variables
x_bike = model.addVar(vtype=GRB.INTEGER, name="x_bike")
x_scooter = model.addVar(vtype=GRB.INTEGER, name="x_scooter")

# Constraints
# 1. Charge capacity
model.addConstr(charge_bike * x_bike + charge_scooter * x_scooter <= total_charge, "charge_limit")

# 2. Bike share limit: 0.30*(x_bike+x_scooter) >= x_bike => 7*x_bike <= 3*x_scooter
model.addConstr(7 * x_bike <= 3 * x_scooter, "bike_share_limit")

# 3. Minimum scooters
model.addConstr(x_scooter >= min_scooters, "min_scooters")

# Objective: maximize meals delivered
model.setObjective(cap_bike * x_bike + cap_scooter * x_scooter, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Save optimal value
if model.Status == GRB.OPTIMAL:
    opt_val = model.ObjVal
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(opt_val))