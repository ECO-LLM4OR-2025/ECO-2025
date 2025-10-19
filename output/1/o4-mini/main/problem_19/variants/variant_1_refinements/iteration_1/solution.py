import gurobipy as gp
from gurobipy import GRB

def main():
    try:
        # Create model with a descriptive name
        model = gp.Model("FurnitureOrder")

        # Suppress console log lines (optional)
        model.setParam('OutputFlag', 0)
        # Ensure reproducibility
        model.setParam('Seed', 1)

        # Parameters
        manufacturers = ['A', 'B', 'C']
        chairs_per_order = {'A': 15, 'B': 10, 'C': 10}
        cost_per_order   = {'A': 15 * 50, 'B': 10 * 45, 'C': 10 * 40}
        L = 100     # minimum chairs
        U = 500     # maximum chairs

        # Precompute maximum possible orders for big-M linking
        max_orders = {
            m: U // chairs_per_order[m] for m in manufacturers
        }

        # Decision variables
        x = {}  # number of orders (integer)
        y = {}  # binary indicator for using a manufacturer
        for m in manufacturers:
            x[m] = model.addVar(vtype=GRB.INTEGER,
                                 name=f"x_{m}")
            y[m] = model.addVar(vtype=GRB.BINARY,
                                 name=f"y_{m}")

        model.update()

        # Objective: minimize total cost
        model.setObjective(
            gp.quicksum(cost_per_order[m] * x[m] for m in manufacturers),
            GRB.MINIMIZE
        )

        # 1) Total chairs constraints
        model.addConstr(
            gp.quicksum(chairs_per_order[m] * x[m] for m in manufacturers) >= L,
            name="MinTotalChairs"
        )
        model.addConstr(
            gp.quicksum(chairs_per_order[m] * x[m] for m in manufacturers) <= U,
            name="MaxTotalChairs"
        )

        # 2) Big-M linking: if y[m]=0 then x[m]=0; if y[m]=1 then x[m]>=1
        for m in manufacturers:
            model.addConstr(x[m] <= max_orders[m] * y[m],
                            name=f"LinkUpper_{m}")
            model.addConstr(x[m] >= y[m],
                            name=f"LinkLower_{m}")

        # 3) Logical implications:
        #    If we order from A then we must order from B (y_A <= y_B)
        #    If we order from B then we must order from C (y_B <= y_C)
        model.addConstr(y['A'] <= y['B'], name="IfAthenB")
        model.addConstr(y['B'] <= y['C'], name="IfBthenC")

        # Optimize the model
        model.optimize()

        # Check for optimal solution and write the objective value
        if model.status == GRB.OPTIMAL:
            opt_val = model.objVal
            try:
                with open('ref_optimal_value.txt', 'w') as f:
                    f.write(f"{opt_val:.0f}")
            except IOError as e:
                print(f"Error writing output file: {e}")
        else:
            print(f"Optimization ended with status {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

if __name__ == "__main__":
    main()