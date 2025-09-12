import gurobipy as gp
from gurobipy import GRB

def main():
    # Create model
    model = gp.Model("scooter_bike_production")

    # Parameters
    profit_scooter = 200
    profit_bike = 300
    design_time_scooter = 2
    design_time_bike = 4
    eng_time_scooter = 3
    eng_time_bike = 5
    avail_design_time = 5000
    avail_eng_time = 6000

    # Decision variables
    x_scooter = model.addVar(vtype=GRB.CONTINUOUS, name="x_scooter", lb=0)
    x_bike = model.addVar(vtype=GRB.CONTINUOUS, name="x_bike", lb=0)

    # Constraints
    model.addConstr(design_time_scooter * x_scooter + design_time_bike * x_bike <= avail_design_time,
                    name="DesignCapacity")
    model.addConstr(eng_time_scooter * x_scooter + eng_time_bike * x_bike <= avail_eng_time,
                    name="EngineeringCapacity")

    # Objective
    model.setObjective(profit_scooter * x_scooter + profit_bike * x_bike, GRB.MAXIMIZE)

    # Optimize
    model.optimize()

    # Save optimal objective value
    if model.status == GRB.OPTIMAL:
        with open('ref_optimal_value.txt', 'w') as f:
            f.write(str(model.objVal))
    else:
        with open('ref_optimal_value.txt', 'w') as f:
            f.write("No optimal solution found.")

if __name__ == "__main__":
    main()