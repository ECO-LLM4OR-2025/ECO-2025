import gurobipy as gp
from gurobipy import GRB

# Data
I = [1, 2, 3]
EarliestLanding = {1: 1, 2: 3, 3: 5}
LatestLanding   = {1:10, 2:12, 3:15}
TargetLanding   = {1: 4, 2: 8, 3:14}
PenaltyBefore   = {1: 5, 2:10, 3:15}
PenaltyAfter    = {1:10, 2:20, 3:30}
SeparationTime  = {
    (1,2): 2, (1,3): 3,
    (2,1): 2, (2,3): 4,
    (3,1): 3, (3,2): 4,
    (1,1): 0, (2,2): 0, (3,3): 0
}

# Create model
m = gp.Model("AircraftLanding")

# Variables
t    = m.addVars(I, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")
earl = m.addVars(I, lb=0.0, name="earliness")
tard = m.addVars(I, lb=0.0, name="tardiness")

# Time window constraints
for i in I:
    m.addConstr(t[i] >= EarliestLanding[i], name=f"earliest_{i}")
    m.addConstr(t[i] <= LatestLanding[i],   name=f"latest_{i}")

# Earliness and tardiness constraints
for i in I:
    m.addConstr(earl[i] >= TargetLanding[i] - t[i], name=f"earl_def_{i}")
    m.addConstr(tard[i] >= t[i] - TargetLanding[i], name=f"tard_def_{i}")

# Separation constraints (i < j)
for i in I:
    for j in I:
        if i < j:
            sep = SeparationTime[(i,j)]
            m.addConstr(t[j] >= t[i] + sep, name=f"sep_{i}_{j}")

# Objective
obj = gp.quicksum(PenaltyBefore[i] * earl[i] + PenaltyAfter[i] * tard[i] for i in I)
m.setObjective(obj, GRB.MINIMIZE)

# Optimize
m.optimize()

# Write optimal value to file
if m.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(m.objVal))