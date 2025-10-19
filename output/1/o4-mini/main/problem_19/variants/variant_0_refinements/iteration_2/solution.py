import gurobipy as gp
from gurobipy import GRB

def main():
    try:
        # -------------------------
        # Data Definition
        # -------------------------
        manufacturers     = ["A", "B", "C"]
        chairs_per_order  = {"A": 15, "B": 10, "C": 10}
        cost_per_chair    = {"A": 50, "B": 45, "C": 40}
        min_chairs = 100
        max_chairs = 500

        # Precompute cost per order and maximum possible orders (Big-M)
        cost_per_order = {m: chairs_per_order[m] * cost_per_chair[m]
                          for m in manufacturers}
        max_orders = {m: max_chairs // chairs_per_order[m]
                      for m in manufacturers}

        # -------------------------
        # Model Setup
        # -------------------------
        model = gp.Model("furniture_ordering")
        # Silence solver output for clean runs
        model.Params.OutputFlag = 0

        # Decision Variables
        # x[m] = number of orders from manufacturer m (integer ≥ 0)
        # y[m] = 1 if we place any order with m, else 0
        x = model.addVars(manufacturers, 
                          vtype=GRB.INTEGER, 
                          lb=0, 
                          name="orders")
        y = model.addVars(manufacturers, 
                          vtype=GRB.BINARY, 
                          name="use_flag")

        # -------------------------
        # Constraints
        # -------------------------
        # 1) Total chairs between min and max
        total_chairs = gp.quicksum(chairs_per_order[m] * x[m] 
                                   for m in manufacturers)
        model.addConstr(total_chairs >= min_chairs, name="min_chairs")
        model.addConstr(total_chairs <= max_chairs, name="max_chairs")

        # 2) Link x and y with linear constraints:
        #    If y[m] = 0 => x[m] = 0; If y[m] = 1 => x[m] ≥ 1
        for m in manufacturers:
            model.addConstr(x[m] <= max_orders[m] * y[m], 
                            name=f"link_up_{m}")
            model.addConstr(x[m] >= y[m], 
                            name=f"link_low_{m}")

        # 3) Logical implications:
        #    If A is used then B must be used (y[A] ≤ y[B])
        #    If B is used then C must be used (y[B] ≤ y[C])
        model.addConstr(y["A"] <= y["B"], name="A_implies_B")
        model.addConstr(y["B"] <= y["C"], name="B_implies_C")

        # -------------------------
        # Objective
        # -------------------------
        total_cost = gp.quicksum(cost_per_order[m] * x[m] 
                                 for m in manufacturers)
        model.setObjective(total_cost, GRB.MINIMIZE)

        # -------------------------
        # Optimize
        # -------------------------
        model.optimize()

        # -------------------------
        # Process Solution
        # -------------------------
        if model.Status == GRB.OPTIMAL:
            obj_val = int(model.ObjVal)
            try:
                with open("ref_optimal_value.txt", "w") as fout:
                    fout.write(str(obj_val))
            except IOError as file_err:
                print(f"File write error: {file_err}")
        else:
            print(f"No optimal solution found. Status code: {model.Status}")

    except gp.GurobiError as gu_err:
        print(f"GurobiError encountered: {gu_err}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

if __name__ == "__main__":
    main()