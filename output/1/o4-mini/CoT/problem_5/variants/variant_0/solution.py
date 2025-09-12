# code starts
import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("flooring_production")

# Decision variables
H = model.addVar(lb=0, name="Hardwood")  # Hardwood sq ft
V = model.addVar(lb=0, name="Vinyl")     # Vinyl sq ft

# Constraints
model.addConstr(H >= 20000, name="MinHardwood")
model.addConstr(V >= 10000, name="MinVinyl")
model.addConstr(H + V >= 60000, name="MinTotalShip")
model.addConstr(H <= 50000, name="MaxHardwood")
model.addConstr(V <= 30000, name="MaxVinyl")

# Objective: maximize profit
model.setObjective(2.5 * H + 3 * V, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Save optimal objective value
with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(model.objVal))
# code ends