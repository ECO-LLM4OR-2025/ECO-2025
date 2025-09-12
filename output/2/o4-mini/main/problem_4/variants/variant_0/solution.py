import gurobipy as gp

def main():
    # Create model
    m = gp.Model("farm_profit")

    # Decision variables
    cows = m.addVar(vtype=gp.GRB.INTEGER, name="cows")
    sheep = m.addVar(vtype=gp.GRB.INTEGER, name="sheep")
    chickens = m.addVar(vtype=gp.GRB.INTEGER, name="chickens")

    # Constraints
    m.addConstr(10 * cows + 5 * sheep + 3 * chickens <= 800, "manure_capacity")
    m.addConstr(chickens <= 50, "max_chickens")
    m.addConstr(cows >= 10, "min_cows")
    m.addConstr(sheep >= 20, "min_sheep")
    m.addConstr(cows + sheep + chickens <= 100, "total_animals")

    # Objective: maximize profit
    m.setObjective(400 * cows + 120 * sheep + 3 * chickens, gp.GRB.MAXIMIZE)

    # Solve
    m.optimize()

    # Write optimal value to file
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(m.objVal))

if __name__ == "__main__":
    main()