import gurobipy as gp
from gurobipy import GRB, GurobiError

def main():
    try:
        # 1. Create the model with a descriptive name
        model = gp.Model("ResourceAllocation")

        # 2. Define decision variables with bounds and integrality
        # x_X: units allocated to project X, between 0 and 700
        xX = model.addVar(vtype=GRB.INTEGER, lb=0, ub=700, name="x_X")
        # x_Y: units allocated to project Y, between 0 and 500
        xY = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="x_Y")

        # 3. Integrate new variables into the model
        model.update()

        # 4. Add constraints
        # Total resource constraint: x_X + x_Y <= 1000
        model.addConstr(xX + xY <= 1000, name="TotalCapacity")
        # Minimum excess constraint: x_X - x_Y >= 200
        model.addConstr(xX - xY >= 200, name="MinExcess")

        # 5. Set objective: minimize total cost 50*x_X + 30*x_Y
        model.setObjective(50 * xX + 30 * xY, GRB.MINIMIZE)

        # 6. Optimize the model
        model.optimize()

        # 7. Check solution status
        if model.Status == GRB.OPTIMAL:
            # Round the objective to the nearest dollar
            optimal_cost = round(model.ObjVal)
            # 8. Write only the optimal value to file
            with open('ref_optimal_value.txt', 'w') as f:
                f.write(f"{optimal_cost}")
        else:
            print(f"Optimization ended with status {model.Status}, no optimal solution found.")

    except GurobiError as e:
        # Handle Gurobi-specific errors
        print(f"Gurobi Error: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()