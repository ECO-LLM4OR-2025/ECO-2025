import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("meal_delivery")

# Parameters
cap_bike = 8
cap_scooter = 5
chg_bike = 3
chg_scooter = 2
C_total = 200
# From  x_bike ≤ 0.3*(x_bike + x_scooter)  => 7 x_bike ≤ 3 x_scooter
y_min = 20

# Decision variables
x_b = model.addVar(vtype=GRB.INTEGER, lb=0, name="bikes")
x_s = model.addVar(vtype=GRB.INTEGER, lb=0, name="scooters")

# Objective: maximize meals delivered
model.setObjective(cap_bike * x_b + cap_scooter * x_s, GRB.MAXIMIZE)

# Constraints
# 1. Charge budget
model.addConstr(chg_bike * x_b + chg_scooter * x_s <= C_total, name="charge_limit")

# 2. Bike share limit: 7*x_b ≤ 3*x_s
model.addConstr(7 * x_b <= 3 * x_s, name="bike_share_limit")

# 3. Minimum scooters
model.addConstr(x_s >= y_min, name="min_scooters")

# Optimize
model.optimize()

# Write out the optimal objective value
if model.status == GRB.OPTIMAL:
    opt_val = model.ObjVal
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(opt_val))
else:
    # In case no solution found, still write something
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("No optimal solution found.")