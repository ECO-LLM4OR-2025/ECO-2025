import gurobipy as gp
from gurobipy import GRB

# Create model
m = gp.Model("resource_allocation")

# Decision variables
x = m.addVar(vtype=GRB.INTEGER, lb=0, ub=700, name="x")   # allocation to project X
y = m.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="y")   # allocation to project Y

# Constraints
m.addConstr(x + y <= 1000, name="total_resource")
m.addConstr(x - y >= 200, name="excess_requirement")

# Objective: minimize cost
m.setObjective(50 * x + 30 * y, GRB.MINIMIZE)

# Solve
m.optimize()

# Write optimal value to file
if m.status == GRB.OPTIMAL:
    opt_val = m.objVal
    with open("ref_optimal_value.txt", "w") as f:
        f.write(f"{opt_val:.0f}")