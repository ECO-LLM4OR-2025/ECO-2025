import sys

# 1. Ensure Gurobi is available
try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    sys.exit("Error: gurobipy not found. Please install Gurobi Python API (gurobipy).")

# 2. Problem data
jobs = [1, 2, 3]
machines = [1, 2]
proces_time = {
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}
# Big-M bound: sum of all processing times (safe upper bound on makespan)
Mbig = sum(proces_time.values())

try:
    # 3. Build Gurobi model
    model = gp.Model("flowshop_2machine_3jobs")
    # Suppress solver output, set a time limit
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 30

    # 4. Decision variables
    # S[j,m]: start time of job j on machine m
    S = model.addVars(jobs, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="S")
    # y[i,j] for i<j: 1 if job i precedes job j on the machines, 0 otherwise
    y_pairs = [(i, j) for i in jobs for j in jobs if i < j]
    y = model.addVars(y_pairs, vtype=GRB.BINARY, name="y")
    # Cmax: makespan
    Cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

    # 5. Constraints

    # 5.1 Machine capacity constraints for machine 1 and 2
    #    We enforce for each pair (i<j) two alternatives:
    #    if y[i,j]=1 then i before j; else j before i.
    for (i, j) in y_pairs:
        # On machine 1
        model.addConstr(
            S[i, 1] + proces_time[(i, 1)]
            <= S[j, 1] + Mbig * (1 - y[i, j]),
            name=f"m1_{i}_{j}_i_before_j"
        )
        model.addConstr(
            S[j, 1] + proces_time[(j, 1)]
            <= S[i, 1] + Mbig * y[i, j],
            name=f"m1_{i}_{j}_j_before_i"
        )
        # On machine 2
        model.addConstr(
            S[i, 2] + proces_time[(i, 2)]
            <= S[j, 2] + Mbig * (1 - y[i, j]),
            name=f"m2_{i}_{j}_i_before_j"
        )
        model.addConstr(
            S[j, 2] + proces_time[(j, 2)]
            <= S[i, 2] + Mbig * y[i, j],
            name=f"m2_{i}_{j}_j_before_i"
        )

    # 5.2 In‐job precedence: a job must finish on machine 1 before starting on machine 2
    for j in jobs:
        model.addConstr(
            S[j, 2] >= S[j, 1] + proces_time[(j, 1)],
            name=f"prec_{j}"
        )

    # 5.3 Makespan definition: Cmax is at least the completion time on machine 2
    for j in jobs:
        model.addConstr(
            Cmax >= S[j, 2] + proces_time[(j, 2)],
            name=f"cmax_def_{j}"
        )

    # 6. Objective: minimize the makespan
    model.setObjective(Cmax, GRB.MINIMIZE)

    # 7. Optimize
    model.optimize()

    # 8. Check for optimality
    if model.Status != GRB.OPTIMAL:
        sys.exit(f"Error: optimization ended with status {model.Status}.")
    optimal_makespan = model.ObjVal

    # 9. Write only the numeric optimal value to file
    with open("ref_optimal_value.txt", "w") as out_file:
        out_file.write(f"{optimal_makespan}")

except gp.GurobiError as e:
    sys.exit(f"Gurobi Error: {e}")