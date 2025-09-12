import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model('scooter_bike_production')

# Sets and Parameters
products = ['scooter', 'bike']
profit = {'scooter': 200, 'bike': 300}
design_time = {'scooter': 2, 'bike': 4}
eng_time = {'scooter': 3, 'bike': 5}
avail_design = 5000
avail_engineering = 6000

# Decision Variables
x = model.addVars(products, vtype=GRB.INTEGER, name="x", lb=0)

# Constraints
model.addConstr(gp.quicksum(design_time[p] * x[p] for p in products) <= avail_design, name="DesignCapacity")
model.addConstr(gp.quicksum(eng_time[p] * x[p] for p in products) <= avail_engineering, name="EngineeringCapacity")

# Objective
model.setObjective(gp.quicksum(profit[p] * x[p] for p in products), GRB.MAXIMIZE)

# Solve
model.optimize()

# Save optimal value
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))