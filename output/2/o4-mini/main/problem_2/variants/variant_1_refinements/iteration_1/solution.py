import sys

# 1. Try importing Gurobi; if unavailable, give a clear error message.
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    sys.exit("ERROR: gurobipy module not found. Please install Gurobi and its Python interface.")

# 2. Data definition (no unused parameters)
jobs = [1, 2, 3]                # job IDs
machines = [1, 2]               # machine IDs in processing order
processing_time = {             # processing_time[j, m] = time for job j on machine m
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}

# Derive counts and position indices
J = len(jobs)
M = len(machines)
positions = list(range(1, J + 1))   # sequence positions 1..J

# 3. Create the Gurobi model
model = gp.Model("flowshop_makespan")
model.Params.OutputFlag = 1   # turn on solver output

# 4. Decision variables
# Y[j,k] = 1 if job j is assigned to position k
Y = model.addVars(jobs, positions, vtype=GRB.BINARY, name="Y")
# C[k,m] = completion time of the job in sequence position k on machine m
C = model.addVars(positions, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
# C_max = makespan
C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

# 5. Assignment constraints: each job in exactly one position, each position filled by one job
model.addConstrs((Y.sum(j, "*") == 1 for j in jobs), name="AssignJob")
model.addConstrs((Y.sum("*", k) == 1 for k in positions), name="AssignPos")

# 6. Flow-shop timing constraints
for k in positions:
    for m in machines:
        # linear expression for processing time of the job placed at (k,m)
        proc_expr = gp.quicksum(processing_time[j, m] * Y[j, k] for j in jobs)
        
        # first job on first machine
        if k == 1 and m == 1:
            model.addConstr(C[k, m] >= proc_expr, name=f"Time_k{ k }_m{ m }")
        
        # other positions on machine 1
        elif m == 1:
            model.addConstr(C[k, m] >= C[k - 1, m] + proc_expr, name=f"Time_k{ k }_m{ m }")
        
        # first position on machines >1
        elif k == 1:
            model.addConstr(C[k, m] >= C[k, m - 1] + proc_expr, name=f"Time_k{ k }_m{ m }")
        
        # general case: both machine- and job-precedence
        else:
            model.addConstr(C[k, m] >= C[k, m - 1] + proc_expr, name=f"TimeMach_k{ k }_m{ m }")
            model.addConstr(C[k, m] >= C[k - 1, m] + proc_expr, name=f"TimeJob_k{ k }_m{ m }")

# 7. Makespan definition: C_max >= completion of last position on last machine
model.addConstr(C_max >= C[J, machines[-1]], name="MakespanDef")

# 8. Objective: minimize makespan
model.setObjective(C_max, GRB.MINIMIZE)

# 9. Optimize with error handling
try:
    model.optimize()
except gp.GurobiError as e:
    sys.exit(f"ERROR during optimization: {e}")

# 10. Write the optimal value to file (only the number)
if model.Status == GRB.OPTIMAL:
    try:
        with open('ref_optimal_value.txt', 'w') as f:
            f.write(f"{model.objVal}")
    except IOError as e:
        sys.exit(f"ERROR writing output file: {e}")
else:
    sys.exit(f"Optimization ended with status {model.Status}, no optimal solution found.")