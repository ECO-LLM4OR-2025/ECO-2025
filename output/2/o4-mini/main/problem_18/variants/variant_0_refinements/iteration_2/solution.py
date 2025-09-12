import sys

# Attempt to import Gurobi and exit with a clear message if unavailable
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    sys.exit("ERROR: gurobipy is not installed. Please install the Gurobi Python API and configure your license.")

def main():
    try:
        # ----------------------------
        # Data for the 3-aircraft ALP
        # ----------------------------
        aircraft = [1, 2, 3]
        EarliestLanding = {1: 1, 2: 3, 3: 5}
        LatestLanding   = {1:10, 2:12, 3:15}
        TargetLanding   = {1: 4, 2: 8, 3:14}
        PenaltyBefore   = {1: 5, 2:10, 3:15}
        PenaltyAfter    = {1:10, 2:20, 3:30}
        # Separation time matrix for pairs (i,j) with i<j
        SeparationTime = {
            (1,2): 2,
            (1,3): 3,
            (2,3): 4
        }

        # ----------------------------
        # Model creation
        # ----------------------------
        model = gp.Model("AircraftLanding")
        # Suppress solver output (set to 1 to debug)
        model.setParam("OutputFlag", 0)

        # ----------------------------
        # Decision variables
        # ----------------------------
        # t[i]: the landing time of aircraft i within its window
        t = model.addVars(
            aircraft,
            lb=EarliestLanding,
            ub=LatestLanding,
            name="t"
        )
        # earliness and tardiness variables (non-negative)
        earl = model.addVars(aircraft, lb=0.0, name="earliness")
        tard = model.addVars(aircraft, lb=0.0, name="tardiness")

        # ----------------------------
        # Earliness/Tardiness linearization
        # earl[i] >= TargetLanding[i] - t[i]
        # tard[i] >= t[i] - TargetLanding[i]
        # ----------------------------
        for i in aircraft:
            model.addConstr(
                earl[i] >= TargetLanding[i] - t[i],
                name=f"earliness_def_{i}"
            )
            model.addConstr(
                tard[i] >= t[i] - TargetLanding[i],
                name=f"tardiness_def_{i}"
            )

        # ----------------------------
        # Separation constraints (fixed order i<j)
        # t[j] >= t[i] + separationTime[i,j]
        # ----------------------------
        for (i, j), sep in SeparationTime.items():
            model.addConstr(
                t[j] >= t[i] + sep,
                name=f"separation_{i}_{j}"
            )

        # ----------------------------
        # Objective: Minimize total penalties
        # Σ (PenaltyBefore[i] * earl[i] + PenaltyAfter[i] * tard[i])
        # ----------------------------
        obj = gp.quicksum(
            PenaltyBefore[i] * earl[i] + PenaltyAfter[i] * tard[i]
            for i in aircraft
        )
        model.setObjective(obj, GRB.MINIMIZE)

        # ----------------------------
        # Solve the model
        # ----------------------------
        model.optimize()

        # ----------------------------
        # Check solution status
        # ----------------------------
        if model.Status == GRB.OPTIMAL:
            # Write the optimal objective value only
            with open('ref_optimal_value.txt', 'w') as out_file:
                out_file.write(f"{model.ObjVal}")
        else:
            sys.exit(f"ERROR: Model did not solve to optimality (status {model.Status}).")

    except gp.GurobiError as e:
        sys.exit(f"ERROR: Gurobi exception encountered: {e}")
    except Exception as e:
        sys.exit(f"ERROR: Unexpected exception encountered: {e}")

if __name__ == "__main__":
    main()