import sys

# ----------------------------
# Data definitions
# ----------------------------
jobs = [1, 2, 3]               # Job IDs
machines = [1, 2]              # Machine IDs in processing order
# processing_time[(j, m)] = time to process job j on machine m
processing_time = {
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}

n = len(jobs)
last_machine = machines[-1]
schedules = list(range(1, n + 1))  # Sequence positions (called "schedules" here)

# ----------------------------
# Try to import Gurobi
# ----------------------------
use_gurobi = True
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    use_gurobi = False

# ----------------------------
# Model with Gurobi
# ----------------------------
if use_gurobi:
    model = gp.Model("flowshop_2machine")
    # Silence Gurobi log
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 30

    # Decision variables
    # X[j,s] = 1 if job j is assigned to schedule (position) s
    X = model.addVars(jobs, schedules, vtype=GRB.BINARY, name="X")
    # C[s,m] = completion time of the job in schedule s on machine m
    C = model.addVars(schedules, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    # Each schedule s must have exactly one job
    model.addConstrs(
        (gp.quicksum(X[j, s] for j in jobs) == 1 for s in schedules),
        name="OneJobPerSchedule"
    )
    # Each job j must be scheduled exactly once
    model.addConstrs(
        (gp.quicksum(X[j, s] for s in schedules) == 1 for j in jobs),
        name="OneSchedulePerJob"
    )

    # Machine 1 timing constraints
    for s in schedules:
        expr = gp.quicksum(processing_time[j, 1] * X[j, s] for j in jobs)
        if s == 1:
            # First schedule does not wait on previous on M1
            model.addConstr(C[1, 1] >= expr, name="M1_Sched1")
        else:
            # Subsequent schedule must wait until previous finishes on M1
            model.addConstr(C[s, 1] >= C[s - 1, 1] + expr, name=f"M1_Sched{s}")

    # Machine 2 timing constraints
    for s in schedules:
        expr_m2 = gp.quicksum(processing_time[j, 2] * X[j, s] for j in jobs)
        if s == 1:
            # First schedule on M2 must wait for its M1 completion
            model.addConstr(C[1, 2] >= C[1, 1] + expr_m2, name="M2_Sched1")
        else:
            # (a) same job technological precedence
            model.addConstr(C[s, 2] >= C[s, 1] + expr_m2, name=f"M2_Tech_s{s}")
            # (b) sequencing on M2
            model.addConstr(C[s, 2] >= C[s - 1, 2] + expr_m2, name=f"M2_Seq_s{s}")

    # Objective: minimize makespan on the last machine for the last schedule
    model.setObjective(C[n, last_machine], GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Check solver status
    if model.Status != GRB.OPTIMAL:
        sys.stderr.write(f"Error: Gurobi did not find optimal solution (status {model.Status})\n")
        sys.exit(1)

    makespan = model.ObjVal

# ----------------------------
# Fallback: Model with PuLP + CBC
# ----------------------------
else:
    try:
        import pulp
    except ImportError:
        sys.stderr.write("Error: neither gurobipy nor PuLP is installed. Install one solver to proceed.\n")
        sys.exit(1)

    # Create PuLP problem
    prob = pulp.LpProblem("flowshop_2machine", pulp.LpMinimize)

    # Decision variables
    X = pulp.LpVariable.dicts("X", (jobs, schedules), lowBound=0, upBound=1, cat="Binary")
    C = pulp.LpVariable.dicts("C", (schedules, machines), lowBound=0, cat="Continuous")

    # Objective
    prob += C[n][last_machine]

    # Assignment constraints
    for s in schedules:
        prob += pulp.lpSum(X[j][s] for j in jobs) == 1, f"OneJobPerSchedule_{s}"
    for j in jobs:
        prob += pulp.lpSum(X[j][s] for s in schedules) == 1, f"OneSchedulePerJob_{j}"

    # M1 timing
    for s in schedules:
        expr = pulp.lpSum(processing_time[j, 1] * X[j][s] for j in jobs)
        if s == 1:
            prob += C[1][1] >= expr, "M1_Sched1"
        else:
            prob += C[s][1] >= C[s - 1][1] + expr, f"M1_Sched{s}"

    # M2 timing
    for s in schedules:
        expr_m2 = pulp.lpSum(processing_time[j, 2] * X[j][s] for j in jobs)
        if s == 1:
            prob += C[1][2] >= C[1][1] + expr_m2, "M2_Sched1"
        else:
            prob += C[s][2] >= C[s][1] + expr_m2, f"M2_Tech_s{s}"
            prob += C[s][2] >= C[s - 1][2] + expr_m2, f"M2_Seq_s{s}"

    # Solve quietly with CBC
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    result = prob.solve(solver)

    if result != pulp.LpStatusOptimal:
        sys.stderr.write(f"Error: PuLP did not find optimal solution (status {pulp.LpStatus[result]})\n")
        sys.exit(1)

    makespan = pulp.value(prob.objective)

# ----------------------------
# Write the final makespan
# ----------------------------
try:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(makespan))
except Exception as e:
    sys.stderr.write(f"Error writing result file: {e}\n")
    sys.exit(1)