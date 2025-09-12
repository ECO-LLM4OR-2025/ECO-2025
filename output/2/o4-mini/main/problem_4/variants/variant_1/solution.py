import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("farm_optimization")

# Decision variables: number of cows, sheep, and chickens (integer)
x_c = model.addVar(vtype=GRB.INTEGER, name="cows")
x_s = model.addVar(vtype=GRB.INTEGER, name="sheep")
x_h = model.addVar(vtype=GRB.INTEGER, name="chickens")

# Parameters
profit_c = 400  # 500 - 100
profit_s = 120  # 200 - 80
profit_h = 3    # 8 - 5

# Constraints
model.addConstr(10 * x_c + 5 * x_s + 3 * x_h <= 800, name="ManureCapacity")
model.addConstr(x_h <= 50,              name="MaxChickens")
model.addConstr(x_c >= 10,              name="MinCows")
model.addConstr(x_s >= 20,              name="MinSheep")
model.addConstr(x_c + x_s + x_h <= 100, name="TotalAnimals")

# Objective: maximize total profit
model.setObjective(profit_c * x_c + profit_s * x_s + profit_h * x_h, GRB.MAXIMIZE)

# Optimize
model.optimize()

# Save the optimal objective value to a file
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))