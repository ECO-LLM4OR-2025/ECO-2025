import sys

# 1. Import Gurobi with error handling
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    sys.stderr.write("Error: gurobipy package not found. Please install gurobipy and ensure you have a valid license.\n")
    sys.exit(1)

def main():
    # 2. Problem data
    I = [1, 2, 3]
    EarliestLanding = {1: 1,  2: 3,  3: 5}
    LatestLanding   = {1: 10, 2: 12, 3: 15}
    TargetLanding   = {1: 4,  2: 8,  3: 14}
    PenBefore       = {1: 5,  2: 10, 3: 15}
    PenAfter        = {1: 10, 2: 20, 3: 30}
    SeparationTime  = {
        (1,2): 2, (1,3): 3,
        (2,3): 4
    }

    try:
        # 3. Create model
        model = gp.Model("AircraftLanding")
        # 4. Suppress solver output for clarity
        model.Params.OutputFlag = 0

        # 5. Decision variables
        #    t[i]: landing time, bounded by earliest/latest directly
        t = model.addVars(
            I,
            lb={i: EarliestLanding[i] for i in I},
            ub={i: LatestLanding[i]   for i in I},
            name="t"
        )
        #    e[i], l[i]: early/late deviations
        e = model.addVars(I, lb=0.0, name="e")
        l = model.addVars(I, lb=0.0, name="l")

        # 6. Separation constraints (fixed sequence 1→2→3)
        for (i, j), sep in SeparationTime.items():
            model.addConstr(t[j] - t[i] >= sep, name=f"sep_{i}_{j}")

        # 7. Deviation constraints: e ≥ Target - t, l ≥ t - Target
        for i in I:
            model.addConstr(e[i] >= TargetLanding[i] - t[i], name=f"early_dev_{i}")
            model.addConstr(l[i] >= t[i] - TargetLanding[i], name=f"late_dev_{i}")

        # 8. Objective: minimize total penalty
        obj = gp.quicksum(PenBefore[i]*e[i] + PenAfter[i]*l[i] for i in I)
        model.setObjective(obj, GRB.MINIMIZE)

        # 9. Optimize
        model.optimize()

    except gp.GurobiError as ge:
        sys.stderr.write(f"Gurobi error: {ge}\n")
        sys.exit(1)

    # 10. Check for optimality and write result
    if model.Status == GRB.OPTIMAL:
        try:
            with open("ref_optimal_value.txt", "w") as fout:
                fout.write(str(model.ObjVal))
        except IOError as ioe:
            sys.stderr.write(f"File write error: {ioe}\n")
            sys.exit(1)
    else:
        sys.stderr.write("Optimization did not reach an optimal solution.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()