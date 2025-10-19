import gurobipy as gp
from gurobipy import GRB

def main():
    # Create model
    model = gp.Model()

    # Variables: number of orders (integer) and binary indicators
    xA = model.addVar(vtype=GRB.INTEGER, name="xA")
    xB = model.addVar(vtype=GRB.INTEGER, name="xB")
    xC = model.addVar(vtype=GRB.INTEGER, name="xC")
    yA = model.addVar(vtype=GRB.BINARY,  name="yA")
    yB = model.addVar(vtype=GRB.BINARY,  name="yB")
    yC = model.addVar(vtype=GRB.BINARY,  name="yC")

    # Parameters
    aA, aB, aC = 15, 10, 10
    cA, cB, cC = 750, 450, 400
    L, U = 100, 500

    # Constraints
    model.addConstr(aA*xA + aB*xB + aC*xC >= L, name="minChairs")
    model.addConstr(aA*xA + aB*xB + aC*xC <= U, name="maxChairs")

    # Big-M linking between x and y:
    model.addConstr(xA <= (U//aA) * yA, name="link_xA_yA")
    model.addConstr(xB <= (U//aB) * yB, name="link_xB_yB")
    model.addConstr(xC <= (U//aC) * yC, name="link_xC_yC")
    model.addConstr(xA >= yA,               name="min_xA_if_yA")
    model.addConstr(xB >= yB,               name="min_xB_if_yB")
    model.addConstr(xC >= yC,               name="min_xC_if_yC")

    # Logical implications
    model.addConstr(yA <= yB, name="ifA_then_B")
    model.addConstr(yB <= yC, name="ifB_then_C")

    # Objective: minimize total cost
    model.setObjective(cA*xA + cB*xB + cC*xC, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Save optimal value
    if model.status == GRB.OPTIMAL:
        with open('ref_optimal_value.txt', 'w') as f:
            f.write(str(model.objVal))

if __name__ == "__main__":
    main()