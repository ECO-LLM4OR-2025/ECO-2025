import gurobipy as gp
from gurobipy import GRB

# Data
J = [1, 2, 3]               # jobs
M = [1, 2]                  # machines
t = {
    (1,1): 1, (1,2): 3,
    (2,1): 2, (2,2): 2,
    (3,1): 3, (3,2): 1
}
# Big M: sum of all processing times
Mbig = sum(t[jm] for jm in t)

# Model
model = gp.Model("flowshop_2machine_3jobs")

# Variables
S = model.addVars(J, M, lb=0.0, name="S")         # start times
y = model.addVars(J, J, vtype=GRB.BINARY, name="y")# order binaries
Cmax = model.addVar(lb=0.0, name="Cmax")          # makespan

# Constraints
# 1. sequencing consistency
for i in J:
    for j in J:
        if i < j:
            model.addConstr(y[i,j] + y[j,i] == 1, name=f"seq_{i}_{j}")

# 2. machine capacity on machine 1
for i in J:
    for j in J:
        if i != j:
            model.addConstr(
                S[i,1] + t[i,1] <= S[j,1] + Mbig * (1 - y[i,j]),
                name=f"m1_cap_{i}_{j}"
            )

# 3. job precedence across machines
for j in J:
    model.addConstr(
        S[j,2] >= S[j,1] + t[j,1],
        name=f"prec_{j}"
    )

# 4. machine capacity on machine 2
for i in J:
    for j in J:
        if i != j:
            model.addConstr(
                S[i,2] + t[i,2] <= S[j,2] + Mbig * (1 - y[i,j]),
                name=f"m2_cap_{i}_{j}"
            )

# 5. makespan definition
for j in J:
    model.addConstr(
        Cmax >= S[j,2] + t[j,2],
        name=f"cmax_def_{j}"
    )

# Objective
model.setObjective(Cmax, GRB.MINIMIZE)

# Solve
model.optimize()

# Save result
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))