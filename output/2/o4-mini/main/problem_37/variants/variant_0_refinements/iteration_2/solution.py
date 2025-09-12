import sys
import gurobipy as gp
from gurobipy import GRB

# Problem data
shape_time = { "thin": 50, "stubby": 30 }
bake_time  = { "thin": 90, "stubby": 150 }
profit     = { "thin": 5,  "stubby": 9 }
shape_cap  = 3000
bake_cap   = 4000

try:
    # Initialize Gurobi model
    model = gp.Model("TerracottaJars")
except Exception as e:
    sys.exit(f"Error initializing Gurobi model: {e}")

# Silence solver log
model.setParam(GRB.Param.OutputFlag, 0)

# Decision variables: number of thin and stubby jars (integer, ≥0)
x = {}
for p in ["thin", "stubby"]:
    x[p] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{p}")

model.update()

# Resource constraints
# 1) Shaping time capacity
model.addConstr(
    gp.quicksum(shape_time[p] * x[p] for p in x) <= shape_cap,
    name="ShapingCapacity"
)
# 2) Baking time capacity
model.addConstr(
    gp.quicksum(bake_time[p] * x[p] for p in x) <= bake_cap,
    name="BakingCapacity"
)

# Objective: maximize total profit
model.setObjective(
    gp.quicksum(profit[p] * x[p] for p in x),
    GRB.MAXIMIZE
)

# Solve the model
try:
    model.optimize()
except gp.GurobiError as e:
    sys.exit(f"Gurobi optimization error: {e}")

# Check solution status
if model.Status != GRB.OPTIMAL:
    sys.exit(f"Optimization ended with status {model.Status} (not optimal).")

# Retrieve optimal objective value
opt_val = model.ObjVal

# Write only the numeric optimal value to file
try:
    with open("ref_optimal_value.txt", "w") as f:
        # Cast to integer if the result is integral
        if abs(opt_val - round(opt_val)) < 1e-6:
            f.write(str(int(round(opt_val))))
        else:
            f.write(str(opt_val))
except Exception as e:
    sys.exit(f"Error writing output file: {e}")