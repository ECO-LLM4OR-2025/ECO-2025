import gurobipy as gp
from gurobipy import GRB

def main():
    try:
        # Create a new model
        model = gp.Model("resource_allocation")
        
        # Decision variables
        # x_X: allocation to project X (0 ≤ x_X ≤ 700, integer)
        # x_Y: allocation to project Y (0 ≤ x_Y ≤ 500, integer)
        x_X = model.addVar(vtype=GRB.INTEGER, lb=0, ub=700, name="x_X")
        x_Y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="x_Y")
        
        # Update model to integrate new variables
        model.update()
        
        # Objective: minimize total cost = 50*x_X + 30*x_Y
        model.setObjective(50 * x_X + 30 * x_Y, GRB.MINIMIZE)
        
        # Constraints:
        # 1) Total resource constraint: x_X + x_Y ≤ 1000
        # 2) Excess requirement:        x_X - x_Y ≥ 200
        model.addConstr(x_X + x_Y <= 1000, name="total_resource")
        model.addConstr(x_X - x_Y >= 200, name="excess_requirement")
        
        # Optimize the model
        model.optimize()
        
        # Check optimization status
        if model.status == GRB.OPTIMAL:
            # Retrieve the optimal objective value
            optimal_cost = model.objVal
            # Write only the integer cost to file
            with open("ref_optimal_value.txt", "w") as f:
                f.write(f"{int(round(optimal_cost))}")
        else:
            raise Exception(f"Optimization did not reach optimality (status {model.status}).")
    
    except gp.GurobiError as e:
        print(f"Gurobi Error {e.errno}: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()