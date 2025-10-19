import gurobipy as gp
from gurobipy import GRB

# Model
model = gp.Model("furniture_ordering")

# Parameters
chairs = {"A": 15, "B": 10, "C": 10}
cost   = {"A": 50, "B": 45, "C": 40}
min_chairs = 100
max_chairs = 500
M = {i: max_chairs // chairs[i] for i in chairs}

# Decision Variables
x = model.addVars(chairs.keys(), vtype=GRB.INTEGER, name="x")
y = model.addVars(chairs.keys(), vtype=GRB.BINARY,  name="y")

# Constraints
# 1) Total chairs bounds
model.addConstr(gp.quicksum(chairs[i] * x[i] for i in chairs) >= min_chairs, "min_chairs")
model.addConstr(gp.quicksum(chairs[i] * x[i] for i in chairs) <= max_chairs, "max_chairs")

# 2) Link x and y
for i in chairs:
    model.addConstr(x[i] <= M[i] * y[i], name=f"link_up_{i}")
    model.addConstr(x[i] >= y[i],       name=f"link_dn_{i}")

# 3) Logical implications
model.addConstr(x["B"] >= y["A"], name="A_implies_B")
model.addConstr(x["C"] >= y["B"], name="B_implies_C")

# Objective: minimize total cost
model.setObjective(gp.quicksum(cost[i] * chairs[i] * x[i] for i in chairs), GRB.MINIMIZE)

# Optimize
model.optimize()

# Save optimal value
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.ObjVal))