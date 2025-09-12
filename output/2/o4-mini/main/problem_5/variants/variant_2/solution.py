import gurobipy as gp
from gurobipy import GRB

# Create model
m = gp.Model('FlooringProduction')

# Parameters
min_h = 20000      # min demand hardwood
min_v = 10000      # min demand vinyl
min_total = 60000  # min total shipment
max_h = 50000      # max capacity hardwood
max_v = 30000      # max capacity vinyl
profit_h = 2.5     # profit per sq ft hardwood
profit_v = 3.0     # profit per sq ft vinyl

# Decision variables
x_h = m.addVar(lb=0, name='hardwood')
x_v = m.addVar(lb=0, name='vinyl')

# Constraints
m.addConstr(x_h >= min_h, name='min_hardwood')
m.addConstr(x_v >= min_v, name='min_vinyl')
m.addConstr(x_h + x_v >= min_total, name='min_total_shipment')
m.addConstr(x_h <= max_h, name='max_hardwood')
m.addConstr(x_v <= max_v, name='max_vinyl')

# Objective: maximize profit
m.setObjective(profit_h * x_h + profit_v * x_v, GRB.MAXIMIZE)

# Optimize
m.optimize()

# Write optimal objective value to file
if m.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(m.objVal))