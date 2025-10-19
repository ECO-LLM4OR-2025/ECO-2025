import gurobipy as gp
from gurobipy import GRB

# Problem parameters
COST_X            = 50    # cost per unit for project X ($)
COST_Y            = 30    # cost per unit for project Y ($)
RESOURCE_CAPACITY = 1000  # total available resource units
EXCESS_REQUIRE    = 200   # x_X - x_Y must be at least this much
MAX_X             = 700   # maximum units assignable to X
MAX_Y             = 500   # maximum units assignable to Y
OUTPUT_FILE       = 'ref_optimal_value.txt'

def main():
    try:
        # Create a Gurobi environment and model in a context manager for safe cleanup
        with gp.Env() as env:
            env.setParam('OutputFlag', 0)   # turn off solver console output
            env.setParam('MIPGap', 0.0)     # require zero optimality gap

            model = gp.Model('ResourceAllocation', env=env)

            # Decision variables: integer allocations for X and Y
            x_X = model.addVar(vtype=GRB.INTEGER, lb=0, ub=MAX_X, name='x_X')
            x_Y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=MAX_Y, name='x_Y')

            # Constraints
            # 1) Total allocation cannot exceed the capacity
            model.addConstr(x_X + x_Y <= RESOURCE_CAPACITY,
                            name='capacity_constr')
            # 2) Project X must exceed project Y by at least EXCESS_REQUIRE
            model.addConstr(x_X - x_Y >= EXCESS_REQUIRE,
                            name='excess_constr')

            # Objective: Minimize total cost
            model.setObjective(COST_X * x_X + COST_Y * x_Y, GRB.MINIMIZE)

            # Ensure all updates are registered before optimizing
            model.update()

            # Optimize the model
            model.optimize()

            # Check for optimality
            if model.Status != GRB.OPTIMAL:
                raise gp.GurobiError(f'Expected OPTIMAL status, got {model.Status}')

            # Round the objective to nearest dollar
            optimal_cost = int(model.ObjVal + 0.5)

        # Write the result to file (only the number, no extra text)
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(str(optimal_cost))

    except gp.GurobiError as gbe:
        print(f'Gurobi Error: {gbe}')
    except Exception as e:
        print(f'Unexpected Error: {e}')

if __name__ == '__main__':
    main()