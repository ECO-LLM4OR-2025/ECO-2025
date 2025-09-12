import gurobipy as gp
from gurobipy import GRB

# Initialize model
model = gp.Model("farm_profit_maximization")

# Decision variables
cows    = model.addVar(lb=0, vtype=GRB.INTEGER, name="cows")
sheep   = model.addVar(lb=0, vtype=GRB.INTEGER, name="sheep")
chickens= model.addVar(lb=0, vtype=GRB.INTEGER, name="chickens")

# Parameters (profits)
profit_cow   = 500 - 100   # 400
profit_sheep = 200 - 80    # 120
profit_chick = 8   - 5     # 3

# Constraints
# 1. Manure capacity
model.addConstr(10*cows + 5*sheep + 3*chickens <= 800, name="manure_cap")
# 2. Chicken capacity
model.addConstr(chickens <= 50, name="chicken_cap")
# 3. Minimum cows
model.addConstr(cows >= 10, name="min_cows")
# 4. Minimum sheep
model.addConstr(sheep >= 20, name="min_sheep")
# 5. Total animals
model.addConstr(cows + sheep + chickens <= 100, name="total_animals")

# Objective: maximize profit
model.setObjective(profit_cow*cows + profit_sheep*sheep + profit_chick*chickens,
                   GRB.MAXIMIZE)

# Solve
model.optimize()

# Write optimal value to file
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))
else:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("No optimal solution found.")