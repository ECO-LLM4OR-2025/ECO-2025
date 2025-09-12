import gurobipy as gp
from gurobipy import GRB

# Create the optimization model
model = gp.Model("scooter_bike_production")

# Decision variables: number of scooters and bikes to produce
x_scooter = model.addVar(vtype=GRB.INTEGER, name="x_scooter", lb=0)
x_bike    = model.addVar(vtype=GRB.INTEGER, name="x_bike", lb=0)

# Parameters
profit_scooter = 200
profit_bike    = 300
design_scooter = 2
design_bike    = 4
eng_scooter    = 3
eng_bike       = 5
design_cap     = 5000
eng_cap        = 6000

# Objective: maximize total profit
model.setObjective(
    profit_scooter * x_scooter + profit_bike * x_bike,
    GRB.MAXIMIZE
)

# Constraints
model.addConstr(design_scooter * x_scooter + design_bike * x_bike <= design_cap,
                name="design_capacity")
model.addConstr(eng_scooter * x_scooter + eng_bike * x_bike <= eng_cap,
                name="engineering_capacity")

# Optimize the model
model.optimize()

# Write the optimal objective value to a file
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))