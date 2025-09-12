import gurobipy as gp
from gurobipy import GRB

# Data
E = {1: 1, 2: 3, 3: 5}
L = {1: 10, 2: 12, 3: 15}
T = {1: 4, 2: 8, 3: 14}
Pb = {1: 5, 2: 10, 3: 15}
Pa = {1: 10, 2: 20, 3: 30}
S = {
    (1,2): 2, (1,3): 3,
    (2,3): 4
}

# Model
model = gp.Model("AircraftLanding")

# Decision variables
t = model.addVars([1,2,3], lb=-gp.GRB.INFINITY, name="t")
e = model.addVars([1,2,3], lb=0.0, name="e")
l = model.addVars([1,2,3], lb=0.0, name="l")

# Time window constraints
for i in [1,2,3]:
    model.addConstr(t[i] >= E[i], name=f"earliest_{i}")
    model.addConstr(t[i] <= L[i], name=f"latest_{i}")

# Separation constraints
for i, j in S:
    model.addConstr(t[j] - t[i] >= S[(i,j)], name=f"sep_{i}_{j}")

# Deviation constraints
for i in [1,2,3]:
    model.addConstr(e[i] >= T[i] - t[i], name=f"early_dev_{i}")
    model.addConstr(l[i] >= t[i] - T[i], name=f"late_dev_{i}")

# Objective
model.setObjective(
    gp.quicksum(Pb[i]*e[i] + Pa[i]*l[i] for i in [1,2,3]),
    GRB.MINIMIZE
)

# Solve
model.optimize()

# Save optimal value
if model.status == GRB.OPTIMAL:
    with open("ref_optimal_value.txt", "w") as f:
        f.write(str(model.objVal))