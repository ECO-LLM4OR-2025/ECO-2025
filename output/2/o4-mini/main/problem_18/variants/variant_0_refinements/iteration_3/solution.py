import sys

# Attempt to import Gurobi; if unavailable, fall back to PuLP (CBC)
USE_GUROBI = True
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    USE_GUROBI = False
    try:
        import pulp as pl
    except ImportError:
        sys.exit("ERROR: Neither gurobipy nor PuLP is installed. Please install one solver.")

def write_result(value):
    """Write only the numeric result to the output file."""
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(f"{value}")

def solve_with_gurobi():
    """Build and solve the ALP using Gurobi."""
    # Data
    aircraft      = [1, 2, 3]
    Earliest      = {1:1, 2:3, 3:5}
    Latest        = {1:10,2:12,3:15}
    Target        = {1:4, 2:8, 3:14}
    PenBefore     = {1:5, 2:10, 3:15}
    PenAfter      = {1:10,2:20,3:30}
    Separation    = {(1,2):2, (1,3):3, (2,3):4}

    # Model
    model = gp.Model("AircraftLanding")
    model.setParam("OutputFlag", 0)    # suppress Gurobi output

    # Variables
    t    = model.addVars(aircraft, lb=Earliest, ub=Latest, name="t")
    earl = model.addVars(aircraft, lb=0.0, name="earliness")
    tard = model.addVars(aircraft, lb=0.0, name="tardiness")

    # Earliness/Tardiness definitions
    for i in aircraft:
        model.addConstr(earl[i] >= Target[i] - t[i], name=f"earl_def_{i}")
        model.addConstr(tard[i] >= t[i] - Target[i], name=f"tard_def_{i}")

    # Separation constraints (fixed order 1<2<3)
    for (i,j), sep in Separation.items():
        model.addConstr(t[j] >= t[i] + sep, name=f"sep_{i}_{j}")

    # Objective
    obj = gp.quicksum(PenBefore[i] * earl[i] + PenAfter[i] * tard[i]
                      for i in aircraft)
    model.setObjective(obj, GRB.MINIMIZE)

    # Solve
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        sys.exit(f"ERROR: Gurobi did not reach optimality (status {model.Status})")

    # Write result
    write_result(model.ObjVal)

def solve_with_pulp():
    """Build and solve the ALP using PuLP + CBC."""
    # Data
    aircraft      = [1, 2, 3]
    Earliest      = {1:1,  2:3,  3:5}
    Latest        = {1:10, 2:12, 3:15}
    Target        = {1:4,  2:8,  3:14}
    PenBefore     = {1:5,  2:10, 3:15}
    PenAfter      = {1:10, 2:20, 3:30}
    Separation    = {(1,2):2, (1,3):3, (2,3):4}

    # Problem
    prob = pl.LpProblem("AircraftLanding", pl.LpMinimize)

    # Variables
    t    = pl.LpVariable.dicts("t",    aircraft, lowBound=None, upBound=None, cat="Continuous")
    earl = pl.LpVariable.dicts("earl",  aircraft, lowBound=0,   cat="Continuous")
    tard = pl.LpVariable.dicts("tard",  aircraft, lowBound=0,   cat="Continuous")
    # Apply time windows
    for i in aircraft:
        t[i].lowBound  = Earliest[i]
        t[i].upBound   = Latest[i]

    # Objective
    prob += pl.lpSum(PenBefore[i]*earl[i] + PenAfter[i]*tard[i] for i in aircraft)

    # Constraints: earliness/tardiness definitions
    for i in aircraft:
        prob += earl[i] >= Target[i] - t[i], f"earl_def_{i}"
        prob += tard[i] >= t[i] - Target[i], f"tard_def_{i}"

    # Separation constraints
    for (i,j), sep in Separation.items():
        prob += t[j] >= t[i] + sep, f"sep_{i}_{j}"

    # Solve with CBC silently
    solver = pl.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pl.LpStatus[status] != "Optimal":
        sys.exit(f"ERROR: PuLP did not reach optimality (status {pl.LpStatus[status]})")

    write_result(pl.value(prob.objective))

if __name__ == "__main__":
    if USE_GUROBI:
        solve_with_gurobi()
    else:
        solve_with_pulp()