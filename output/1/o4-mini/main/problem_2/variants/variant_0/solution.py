import gurobipy as gp
from gurobipy import GRB

# Data
jobs = [1, 2, 3]
machines = [1, 2]
positions = [1, 2, 3]
p = {
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}

# Model
model = gp.Model("flowshop_2machine")

# Decision variables
x = model.addVars(jobs, positions, vtype=GRB.BINARY, name="x")
C = model.addVars(positions, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

# Assignment constraints
for j in jobs:
    model.addConstr(gp.quicksum(x[j, k] for k in positions) == 1,
                    name=f"assign_job_{j}")
for k in positions:
    model.addConstr(gp.quicksum(x[j, k] for j in jobs) == 1,
                    name=f"assign_pos_{k}")

# Flow constraints
for k in positions:
    for m in machines:
        # flow from previous position on same machine
        if k == 1:
            model.addConstr(
                C[k, m] >= gp.quicksum(p[j, m] * x[j, k] for j in jobs),
                name=f"flow_pos{1}_mach{m}"
            )
        else:
            model.addConstr(
                C[k, m] >= C[k-1, m] + gp.quicksum(p[j, m] * x[j, k] for j in jobs),
                name=f"flow_pos{k}_mach{m}"
            )
        # flow from same position on previous machine
        if m > 1:
            model.addConstr(
                C[k, m] >= C[k, m-1] + gp.quicksum(p[j, m] * x[j, k] for j in jobs),
                name=f"flow_pos{k}_mach{m-1}_to_{m}"
            )

# Objective: minimize makespan on last position and last machine
model.setObjective(C[3, 2], GRB.MINIMIZE)

# Optimize
model.optimize()

# Save optimal value
opt_val = model.objVal if model.Status == GRB.OPTIMAL else None
with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(opt_val))