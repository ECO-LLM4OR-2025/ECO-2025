# Try to import Gurobi; if unavailable, fall back to PuLP/CBC
try:
    import gurobipy as gp
    from gurobipy import GRB
    solver_name = "Gurobi"
except ModuleNotFoundError:
    from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, LpStatus, value, PULP_CBC_CMD
    solver_name = "PuLP"

# Problem data (flow‐shop with 3 jobs, 2 machines)
jobs = [1, 2, 3]          # Set J
machines = [1, 2]         # Set M
positions = [1, 2, 3]     # Set K (sequence positions)
# Processing times p[j,m] corresponds to narrative 'proces_time'
p = {
    (1, 1): 1, (1, 2): 3,
    (2, 1): 2, (2, 2): 2,
    (3, 1): 3, (3, 2): 1
}

if solver_name == "Gurobi":
    # Build Gurobi model
    model = gp.Model("flowshop_2machine")
    # Decision variables
    x = model.addVars(jobs, positions, vtype=GRB.BINARY, name="x")      # x[j,k]=1 if job j is in position k
    C = model.addVars(positions, machines, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
    # Assignment constraints: each job assigned to exactly one position, and each position filled
    for j in jobs:
        model.addConstr(gp.quicksum(x[j, k] for k in positions) == 1,
                        name=f"assign_job_{j}")
    for k in positions:
        model.addConstr(gp.quicksum(x[j, k] for j in jobs) == 1,
                        name=f"assign_pos_{k}")
    # Flow‐shop constraints
    # - sequencing on each machine
    # - precedence between machines
    for k in positions:
        for m in machines:
            # Completion on machine m must be at least its own processing
            # plus finish of previous position on same machine
            if k == 1:
                # For the first position, no prior job on this machine
                model.addConstr(
                    C[k, m] >= gp.quicksum(p[j, m] * x[j, k] for j in jobs),
                    name=f"flow_pos{k}_mach{m}"
                )
            else:
                model.addConstr(
                    C[k, m] >= C[k-1, m] + gp.quicksum(p[j, m] * x[j, k] for j in jobs),
                    name=f"flow_pos{k}_mach{m}"
                )
            # Ensure machine precedence: machine m cannot start before job finishes on machine m-1
            if m > 1:
                model.addConstr(
                    C[k, m] >= C[k, m-1] + gp.quicksum(p[j, m] * x[j, k] for j in jobs),
                    name=f"flow_pos{k}_mach{m-1}_to_{m}"
                )
    # Objective: minimize makespan (completion time of last job, last machine)
    last_k = max(positions)
    last_m = max(machines)
    model.setObjective(C[last_k, last_m], GRB.MINIMIZE)
    model.Params.OutputFlag = 1  # enable solver output

    # Solve
    model.optimize()
    # Check status
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution: status {model.Status}")
    opt_val = model.ObjVal

else:
    # Build PuLP model as fallback
    prob = LpProblem("flowshop_2machine", LpMinimize)
    # Variables
    x = LpVariable.dicts("x", (jobs, positions), cat=LpBinary)
    C = LpVariable.dicts("C", (positions, machines), lowBound=0)
    # Assignment constraints
    for j in jobs:
        prob += (sum(x[j][k] for k in positions) == 1), f"assign_job_{j}"
    for k in positions:
        prob += (sum(x[j][k] for j in jobs) == 1), f"assign_pos_{k}"
    # Flow constraints
    for k in positions:
        for m in machines:
            if k == 1:
                prob += (C[k][m] >= sum(p[j,m] * x[j][k] for j in jobs)), f"flow_pos{k}_mach{m}"
            else:
                prob += (C[k][m] >= C[k-1][m] + sum(p[j,m] * x[j][k] for j in jobs)), f"flow_pos{k}_mach{m}"
            if m > 1:
                prob += (C[k][m] >= C[k][m-1] + sum(p[j,m] * x[j][k] for j in jobs)), \
                        f"flow_pos{k}_mach{m-1}_to_{m}"
    # Objective
    last_k = max(positions)
    last_m = max(machines)
    prob += C[last_k][last_m]
    # Solve with CBC
    solver = PULP_CBC_CMD(msg=True)
    result = prob.solve(solver)
    if LpStatus[result] != "Optimal":
        raise RuntimeError(f"PuLP solver did not find optimal solution: status {LpStatus[result]}")
    opt_val = value(prob.objective)

# Write result to file
with open('ref_optimal_value.txt', 'w') as f:
    f.write(f"{opt_val}")