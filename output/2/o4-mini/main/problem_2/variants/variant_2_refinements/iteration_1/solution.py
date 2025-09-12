import sys

# Attempt to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    sys.stderr.write("Error: gurobipy is not installed. "
                     "Please install Gurobi's Python API (e.g., `pip install gurobipy`).\n")
    sys.exit(1)

# Data definition
jobs = [1, 2, 3]                # Job IDs
machines = [1, 2]               # Machine indices in series
# Processing times p[j,m] for job j on machine m
processing_time = {
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}
n = len(jobs)                   # Number of jobs
last_machine = machines[-1]     # Index of the final machine

# Create the Gurobi model
model = gp.Model("flowshop_2machines")
model.setParam('OutputFlag', 0)     # Turn off solver output
model.setParam('TimeLimit', 30)     # Example param: 30‐second time limit

# Decision variables
# X[j,p] = 1 if job j is placed in sequence position p
X = model.addVars(jobs, jobs, vtype=GRB.BINARY, name="X")
# C[p,m] = completion time of the job in position p on machine m
C = model.addVars(range(1, n+1), machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

# Assignment constraints: one job per position, each job exactly once
model.addConstrs(
    (X.sum('*', p) == 1 for p in jobs),
    name="OneJobPerPosition"
)
model.addConstrs(
    (X.sum(j, '*') == 1 for j in jobs),
    name="EachJobOnce"
)

# Completion‐time constraints for machine 1
# Position 1
model.addConstr(
    C[1, 1] >= gp.quicksum(processing_time[j, 1] * X[j, 1] for j in jobs),
    name="C_1_1"
)
# Positions 2..n
for p in range(2, n+1):
    model.addConstr(
        C[p, 1] >= C[p-1, 1] +
                    gp.quicksum(processing_time[j, 1] * X[j, p] for j in jobs),
        name=f"C_{p}_1"
    )

# Completion‐time constraints for machine 2
# Position 1 must wait for machine 1 and its own processing
model.addConstr(
    C[1, 2] >= C[1, 1] +
                gp.quicksum(processing_time[j, 2] * X[j, 1] for j in jobs),
    name="C_1_2"
)
# Positions 2..n: both technological and sequencing precedence
for p in range(2, n+1):
    # Technological precedence: same job on m2 after m1
    model.addConstr(
        C[p, 2] >= C[p, 1] +
                    gp.quicksum(processing_time[j, 2] * X[j, p] for j in jobs),
        name=f"C_{p}_2_afterM1"
    )
    # Sequencing on machine 2: cannot start p before p-1 completes on m2
    model.addConstr(
        C[p, 2] >= C[p-1, 2] +
                    gp.quicksum(processing_time[j, 2] * X[j, p] for j in jobs),
        name=f"C_{p}_2_seq"
    )

# Objective: minimize makespan = completion time of last job on last machine
model.setObjective(C[n, last_machine], GRB.MINIMIZE)

# Optimize and handle the result
model.optimize()

if model.status == GRB.OPTIMAL:
    makespan = model.objVal
    # Save only the numeric makespan to file
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(f"{makespan}")
else:
    sys.stderr.write(f"Optimization ended with status {model.status}, "
                     "no optimal solution found.\n")
    sys.exit(1)