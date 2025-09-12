import sys

# Try importing Gurobi; if unavailable, fall back to PuLP
try:
    import gurobipy as gp
    from gurobipy import GRB
    SOLVER = 'gurobi'
except ModuleNotFoundError:
    try:
        from pulp import LpProblem, LpVariable, LpMaximize, value, PULP_CBC_CMD
        SOLVER = 'pulp'
    except ModuleNotFoundError:
        sys.exit("Error: Neither Gurobi nor PuLP is available. Please install one of these solvers.")

# Model parameters
params = {
    'profit_h': 2.5,    # profit per sqft hardwood
    'profit_v': 3.0,    # profit per sqft vinyl
    'demand_h': 20000,  # min demand hardwood
    'demand_v': 10000,  # min demand vinyl
    'ship_min': 60000,  # min total shipping
    'cap_h': 50000,     # max capacity hardwood
    'cap_v': 30000      # max capacity vinyl
}

# Solve with Gurobi
if SOLVER == 'gurobi':
    # Create Gurobi model
    model = gp.Model("flooring_production")
    model.Params.OutputFlag = 0  # Turn off solver output for clarity

    # Decision variables: x_h and x_v
    x = model.addVars(
        ['hardwood', 'vinyl'],
        lb=0.0,
        name="production"
    )

    # Demand constraints
    model.addConstr(x['hardwood'] >= params['demand_h'], name="hardwood_demand")
    model.addConstr(x['vinyl']   >= params['demand_v'], name="vinyl_demand")

    # Shipping requirement
    model.addConstr(x['hardwood'] + x['vinyl'] >= params['ship_min'], name="total_shipping")

    # Capacity constraints
    model.addConstr(x['hardwood'] <= params['cap_h'], name="hardwood_capacity")
    model.addConstr(x['vinyl']   <= params['cap_v'], name="vinyl_capacity")

    # Objective: maximize total profit
    model.setObjective(
        params['profit_h'] * x['hardwood'] +
        params['profit_v'] * x['vinyl'],
        GRB.MAXIMIZE
    )

    # Optimize
    model.optimize()

    # Check for optimality
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi optimization ended with status {model.status}")

    optimal_value = model.objVal

# Solve with PuLP (fallback)
else:
    # Create PuLP problem
    model = LpProblem("flooring_production", LpMaximize)

    # Decision variables
    x_h = LpVariable("hardwood_sqft", lowBound=0)
    x_v = LpVariable("vinyl_sqft", lowBound=0)

    # Objective
    model += params['profit_h'] * x_h + params['profit_v'] * x_v

    # Constraints
    model += x_h >= params['demand_h'], "hardwood_demand"
    model += x_v >= params['demand_v'], "vinyl_demand"
    model += x_h + x_v >= params['ship_min'], "total_shipping"
    model += x_h <= params['cap_h'], "hardwood_capacity"
    model += x_v <= params['cap_v'], "vinyl_capacity"

    # Solve using the default CBC solver
    result_status = model.solve(PULP_CBC_CMD(msg=False))
    if result_status != 1:  # 1 corresponds to LpStatusOptimal
        raise RuntimeError(f"PuLP optimization ended with status {result_status}")

    optimal_value = value(model.objective)

# Write only the numeric optimal value to file
with open('ref_optimal_value.txt', 'w') as f:
    f.write(f"{optimal_value}")