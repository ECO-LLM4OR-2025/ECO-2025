import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("Elm_Furniture_Production")

# Decision variables
x_chair = model.addVar(vtype=GRB.INTEGER, name="x_chair", lb=0)
x_dresser = model.addVar(vtype=GRB.INTEGER, name="x_dresser", lb=0)

# Parameters
profit_chair = 43
profit_dresser = 52
stain_chair = 1.4
stain_dresser = 1.1
wood_chair = 2
wood_dresser = 3
avail_stain = 17
avail_wood = 11

# Constraints
model.addConstr(stain_chair * x_chair + stain_dresser * x_dresser <= avail_stain, 
                name="StainLimit")
model.addConstr(wood_chair * x_chair + wood_dresser * x_dresser <= avail_wood, 
                name="WoodLimit")

# Objective
model.setObjective(profit_chair * x_chair + profit_dresser * x_dresser, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Save optimal objective value
if model.Status == GRB.OPTIMAL:
    optimal_value = model.ObjVal
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(optimal_value))
    # Optionally print the solution
    # print(f"Optimal production: chairs={x_chair.X}, dressers={x_dresser.X}, profit={optimal_value}")