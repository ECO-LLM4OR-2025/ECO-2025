import gurobipy as gp
from gurobipy import GRB, GurobiError

def main():
    try:
        # Build model in a context manager for automatic cleanup
        with gp.Model("ResourceAllocation") as model:
            # --- Decision Variables ---
            # xX: integer units for project X, between 0 and 700
            xX = model.addVar(vtype=GRB.INTEGER, lb=0, ub=700, name="xX")
            # xY: integer units for project Y, between 0 and 500
            xY = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="xY")

            # --- Objective ---
            # Minimize total cost: 50 per unit of X plus 30 per unit of Y
            model.setObjective(50 * xX + 30 * xY, GRB.MINIMIZE)

            # --- Constraints ---
            # 1) Total allocation must not exceed 1000 units
            model.addConstr(xX + xY <= 1000, name="TotalCapacity")
            # 2) Project X must exceed Project Y by at least 200 units
            model.addConstr(xX - xY >= 200, name="MinExcess")

            # Solve the model
            model.optimize()

            # Check solver status
            if model.Status != GRB.OPTIMAL:
                raise GurobiError(f"Optimal solution not found (status {model.Status})")

            # Retrieve and round the optimal cost
            optimal_cost = round(model.ObjVal)

        # Save only the numeric cost to the reference file
        with open('ref_optimal_value.txt', 'w') as f:
            f.write(str(optimal_cost))

    except GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()