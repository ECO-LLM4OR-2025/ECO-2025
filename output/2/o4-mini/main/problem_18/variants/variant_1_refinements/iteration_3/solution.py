import sys

# 1. Import Gurobi with error handling
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    sys.stderr.write("Error: gurobipy package not found. Please install gurobipy and ensure you have a valid license.\n")
    sys.exit(1)

def main():
    # 2. Problem data (single‐runway, fixed landing order 1→2→3)
    I = [1, 2, 3]
    EarliestLanding = {1: 1, 2: 3, 3: 5}
    LatestLanding   = {1: 10, 2: 12, 3: 15}
    TargetLanding   = {1: 4, 2: 8, 3: 14}
    PenBefore       = {1: 5, 2: 10, 3: 15}
    PenAfter        = {1: 10, 2: 20, 3: 30}
    # Only necessary separations for fixed order; removed redundant (1,3) constraint
    SeparationTime  = {(1, 2): 2, (2, 3): 4}

    try:
        # 3. Create model
        model = gp.Model("AircraftLanding")
        # Enable solver output for debugging; set to 0 for silent runs
        model.Params.OutputFlag = 1

        # 4. Decision variables
        #    t[i] = landing time within its window
        t = model.addVars(I, lb=EarliestLanding, ub=LatestLanding, name="t")
        #    e[i], l[i] = early/late deviation from target
        e = model.addVars(I, lb=0.0, name="early_dev")
        l = model.addVars(I, lb=0.0, name="late_dev")

        # 5. Link deviations to landing times
        for i in I:
            model.addConstr(e[i] >= TargetLanding[i] - t[i], name=f"c_early_{i}")
            model.addConstr(l[i] >= t[i] - TargetLanding[i], name=f"c_late_{i}")

        # 6. Separation constraints for fixed sequence
        for (i, j), sep in SeparationTime.items():
            model.addConstr(t[j] - t[i] >= sep, name=f"c_sep_{i}_{j}")

        # 7. Objective: minimize total early/late penalties
        obj = gp.quicksum(PenBefore[i] * e[i] + PenAfter[i] * l[i] for i in I)
        model.setObjective(obj, GRB.MINIMIZE)

        # 8. Optimize
        model.optimize()

        # 9. Output handling
        if model.Status == GRB.OPTIMAL:
            with open("ref_optimal_value.txt", "w") as fout:
                fout.write(str(model.ObjVal))
        else:
            sys.stderr.write(f"Optimization did not reach optimality (status {model.Status}).\n")
            sys.exit(1)

    except gp.GurobiError as ge:
        sys.stderr.write(f"Gurobi error: {ge}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()