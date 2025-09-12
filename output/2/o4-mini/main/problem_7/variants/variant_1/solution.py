import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("FurnitureProduction")

# Decision variables
chairs   = model.addVar(name="chairs",   vtype=GRB.CONTINUOUS, lb=0)
dressers = model.addVar(name="dressers", vtype=GRB.CONTINUOUS, lb=0)

# Parameters
profit_chair   = 43
profit_dresser = 52
stain_chair    = 1.4
stain_dresser  = 1.1
wood_chair     = 2
wood_dresser   = 3
stain_avail    = 17
wood_avail     = 11

# Constraints
model.addConstr(stain_chair * chairs + stain_dresser * dressers <= stain_avail, name="StainLimit")
model.addConstr(wood_chair  * chairs + wood_dresser  * dressers <= wood_avail,  name="WoodLimit")

# Objective
model.setObjective(profit_chair * chairs + profit_dresser * dressers, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Save optimal value
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))