# code starts
import gurobipy as gp
from gurobipy import GRB

# Data
jobs      = [1, 2, 3]
schedules = [1, 2, 3]
machines  = [1, 2]
# processing times: proces_time_list[job_index][machine_index]
proces_time_list = [[1, 3],
                    [2, 2],
                    [3, 1]]
# convert to dict p[(j,m)]
proces_time = {}
for idx, j in enumerate(jobs):
    for m_idx, m in enumerate(machines):
        proces_time[(j, m)] = proces_time_list[idx][m_idx]

# Create model
model = gp.Model("flowshop_2machine")

# Decision variables
x  = model.addVars(jobs, schedules, vtype=GRB.BINARY, name="x")
C1 = model.addVars(schedules, lb=0.0, vtype=GRB.CONTINUOUS, name="C1")
C2 = model.addVars(schedules, lb=0.0, vtype=GRB.CONTINUOUS, name="C2")
z  = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Makespan")

# 1) Assignment (permutation) constraints
for k in schedules:
    model.addConstr(gp.quicksum(x[j,k] for j in jobs) == 1, name=f"onejob_at_pos_{k}")
for j in jobs:
    model.addConstr(gp.quicksum(x[j,k] for k in schedules) == 1, name=f"job_{j}_assigned_once")

# 2) Machine 1 timing
# Position 1
model.addConstr(
    C1[1] == gp.quicksum(x[j,1] * proces_time[(j,1)] for j in jobs),
    name="C1_pos1"
)
# Positions 2..n
for k in schedules[1:]:
    model.addConstr(
        C1[k] == C1[k-1] + gp.quicksum(x[j,k] * proces_time[(j,1)] for j in jobs),
        name=f"C1_pos{k}"
    )

# 3) Machine 2 timing (flow constraints)
for k in schedules:
    # processing time on machine 2 of job at pos k
    p2k = gp.quicksum(x[j,k] * proces_time[(j,2)] for j in jobs)
    # it cannot start before its own processing on M2
    model.addConstr(C2[k] >= p2k, name=f"C2_proc_pos{k}")
    # it cannot finish before it has finished on M1
    model.addConstr(C2[k] >= C1[k] + p2k, name=f"C2_after_M1_pos{k}")
    # it cannot finish before machine 2 has finished previous job
    if k > 1:
        model.addConstr(C2[k] >= C2[k-1] + p2k, name=f"C2_flow_pos{k}")

# 4) Makespan
n = len(jobs)
model.addConstr(z >= C2[n], name="makespan_link")

# Objective: minimize makespan
model.setObjective(z, GRB.MINIMIZE)

# Optimize
model.optimize()

# Save optimal makespan value
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(z.X))

# code ends