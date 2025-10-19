import gurobipy as gp
from gurobipy import GRB

def main():
    try:
        # Create the model
        model = gp.Model("FurnitureOrdering")
        # Silence solver output for clean logs
        model.Params.OutputFlag = 0
        # Fix random seed for reproducibility
        model.Params.Seed = 42

        # --- Data Definitions ---
        manufacturers    = ['A', 'B', 'C']
        chairs_per_order = {'A': 15, 'B': 10, 'C': 10}
        cost_per_order   = {'A': 15*50, 'B': 10*45, 'C': 10*40}
        min_chairs       = 100
        max_chairs       = 500
        # Compute the maximum number of orders possible per manufacturer
        max_orders = {m: max_chairs // chairs_per_order[m] for m in manufacturers}

        # --- Decision Variables ---
        # x[m]: integer number of orders from manufacturer m
        x = model.addVars(manufacturers,
                          vtype=GRB.INTEGER,
                          lb=0,
                          ub=max_orders,
                          name="x")
        # y[m]: binary flag, 1 if we order at least one batch from m
        y = model.addVars(manufacturers,
                          vtype=GRB.BINARY,
                          name="y")

        # --- Objective: Minimize total cost ---
        model.setObjective(
            gp.quicksum(cost_per_order[m] * x[m] for m in manufacturers),
            GRB.MINIMIZE
        )

        # --- Constraints ---

        # 1) Total chairs ordered between 100 and 500
        total_chairs = gp.quicksum(chairs_per_order[m] * x[m] for m in manufacturers)
        model.addConstr(total_chairs >= min_chairs, name="MinChairs")
        model.addConstr(total_chairs <= max_chairs, name="MaxChairs")

        # 2) Link x and y with indicator constraints:
        #    If y[m] = 1 ⇒ x[m] ≥ 1
        #    If y[m] = 0 ⇒ x[m] = 0
        for m in manufacturers:
            model.addGenConstrIndicator(y[m], 1, x[m] >= 1,
                                        name=f"Ind_x_ge1_if_y_{m}")
            model.addGenConstrIndicator(y[m], 0, x[m] == 0,
                                        name=f"Ind_x_eq0_if_not_y_{m}")

        # 3) Logical supplier implications:
        #    A ⇒ B  (y[A] ≤ y[B]),  B ⇒ C  (y[B] ≤ y[C])
        model.addConstr(y['A'] <= y['B'], name="IfAthenB")
        model.addConstr(y['B'] <= y['C'], name="IfBthenC")

        # --- Optimize ---
        model.optimize()

        # --- Output ---
        if model.Status == GRB.OPTIMAL:
            optimal_value = int(model.ObjVal)
            try:
                with open('ref_optimal_value.txt', 'w') as fout:
                    fout.write(f"{optimal_value}")
            except IOError as io_err:
                print(f"Error writing output file: {io_err}")
        else:
            print(f"Model did not solve to optimality. Status code: {model.Status}")

    except gp.GurobiError as grb_err:
        print(f"Gurobi Error {grb_err.errno}: {grb_err}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

if __name__ == "__main__":
    main()