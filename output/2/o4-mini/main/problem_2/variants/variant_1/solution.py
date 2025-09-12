import gurobipy as gp
from gurobipy import GRB

# Data
jobs = [0, 1, 2]
positions = [0, 1, 2]
machines = [0, 1]
proces_time = {
    (0, 0): 1, (0, 1): 3,
    (1, 0): 2, (1, 1): 2,
    (2, 0): 3, (2, 1): 1
}
J = len(jobs)
M = len(machines)

# Model
model = gp.Model("flowshop_makespan")

# Variables
Y = model.addVars(jobs, positions, vtype=GRB.BINARY, name="Y")
C = model.addVars(positions, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

# Assignment constraints
for j in jobs:
    model.addConstr(gp.quicksum(Y[j, k] for k in positions) == 1)
for k in positions:
    model.addConstr(gp.quicksum(Y[j, k] for j in jobs) == 1)

# Flow-shop timing on machine 1 (m=0)
# Position 0
model.addConstr(
    C[0, 0] >= gp.quicksum(proces_time[j, 0] * Y[j, 0] for j in jobs)
)
# Positions k=1..J-1
for k in positions[1:]:
    model.addConstr(
        C[k, 0] >= C[k-1, 0] + gp.quicksum(proces_time[j, 0] * Y[j, k] for j in jobs)
    )

# Flow-shop timing for machines m=1..M-1
for m in machines[1:]:
    # Position 0
    model.addConstr(
        C[0, m] >= C[0, m-1] + gp.quicksum(proces_time[j, m] * Y[j, 0] for j in jobs)
    )
    # Positions k=1..J-1
    for k in positions[1:]:
        model.addConstr(
            C[k, m] >= C[k, m-1] + gp.quicksum(proces_time[j, m] * Y[j, k] for j in jobs)
        )
        model.addConstr(
            C[k, m] >= C[k-1, m] + gp.quicksum(proces_time[j, m] * Y[j, k] for j in jobs)
        )

# Makespan definition
model.addConstr(C_max >= C[J-1, M-1])

# Objective
model.setObjective(C_max, GRB.MINIMIZE)

# Optimize
model.optimize()

# Write optimal makespan to file
with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(model.objVal))