import gurobipy as gp
from gurobipy import GRB, GurobiError

def main():
    try:
        # Create the model within a context manager for automatic cleanup
        with gp.Model("ResourceAllocation") as model:
            # Solver settings: exact optimum, no console log
            model.Params.MIPGap = 0.0
            model.Params.LogToConsole = 0

            # Decision variables:
            # xX: units for project X, integer in [0,700]
            # xY: units for project Y, integer in [0,500]
            xX = model.addVar(vtype=GRB.INTEGER, lb=0, ub=700, name="x_X")
            xY = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="x_Y")

            # Objective: minimize total cost = 50*xX + 30*xY
            model.setObjective(50 * xX + 30 * xY, GRB.MINIMIZE)

            # Constraints:
            # 1) Total capacity cannot exceed 1000
            model.addConstr(xX + xY <= 1000, name="TotalCapacity")
            # 2) Project X must exceed Project Y by at least 200 units
            model.addConstr(xX - xY >= 200, name="MinExcess")

            # Optimize the model
            model.optimize()

            # Check for optimality
            if model.Status == GRB.OPTIMAL:
                # Round objective to nearest dollar
                optimal_cost = round(model.ObjVal)
                # Write only the cost to the reference file
                with open('ref_optimal_value.txt', 'w') as f:
                    f.write(str(optimal_cost))
            else:
                raise GurobiError(f"No optimal solution found (status {model.Status})")

    except GurobiError as ge:
        print(f"Gurobi Error: {ge}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()