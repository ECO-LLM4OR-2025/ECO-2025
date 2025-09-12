import sys

# 1. Robust import of Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    sys.exit("ERROR: Gurobi module not found. Please install Gurobi and its Python interface.")

# 2. Problem data (removed unused 'schedules')
jobs = [1, 2, 3]                             # Job identifiers
machines = [1, 2]                            # Machines in processing order
processing_time = {                          # Processing time for (job, machine)
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}
positions = list(range(1, len(jobs) + 1))    # Sequence positions

# 3. Create model
model = gp.Model("flowshop_makespan")
model.Params.OutputFlag = 1   # Enable solver output

# 4. Decision variables
# assign[j,k] = 1 if job j is placed in sequence position k
assign = model.addVars(jobs, positions, vtype=GRB.BINARY, name="assign")
# C[k,m] = completion time at position k on machine m
C = model.addVars(positions, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
# Cmax = makespan
Cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

# 5. Assignment constraints
model.addConstrs((assign.sum(j, '*') == 1 for j in jobs), name="onePosPerJob")
model.addConstrs((assign.sum('*', k) == 1 for k in positions), name="oneJobPerPos")

# 6. Flow-shop timing constraints
for k in positions:
    for m in machines:
        # Build expression for processing time of the job at (k,m)
        pt_expr = gp.quicksum(processing_time[j, m] * assign[j, k] for j in jobs)

        # (a) Sequencing precedence on same machine
        if k > 1:
            model.addConstr(
                C[k, m] >= C[k-1, m] + pt_expr,
                name=f"seq_pos{k}_mac{m}"
            )
        # (b) Flow precedence across machines
        if m > 1:
            model.addConstr(
                C[k, m] >= C[k, m-1] + pt_expr,
                name=f"flow_pos{k}_mac{m}"
            )
        # (c) Initialization for the first operation
        if k == 1 and m == 1:
            model.addConstr(
                C[k, m] >= pt_expr,
                name="start_op"
            )

# 7. Makespan definition
last_k = positions[-1]
last_m = machines[-1]
model.addConstr(Cmax >= C[last_k, last_m], name="define_makespan")

# 8. Objective: minimize makespan
model.setObjective(Cmax, GRB.MINIMIZE)

# 9. Optimize with error handling
try:
    model.optimize()
except gp.GurobiError as e:
    sys.exit(f"ERROR during optimization: {e}")

# 10. Output the optimal makespan
if model.Status == GRB.OPTIMAL:
    try:
        with open('ref_optimal_value.txt', 'w') as outfile:
            outfile.write(str(model.objVal))
    except IOError as e:
        sys.exit(f"ERROR writing output file: {e}")
else:
    sys.exit(f"Optimization ended with status {model.Status}, no optimal solution found.")