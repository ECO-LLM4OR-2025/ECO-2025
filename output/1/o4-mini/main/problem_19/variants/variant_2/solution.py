import gurobipy as gp
from gurobipy import GRB

# Data
manufacturers = ['A', 'B', 'C']
c = {'A': 750, 'B': 450, 'C': 400}
s = {'A': 15,  'B': 10,  'C': 10}
min_chairs = 100
max_chairs = 500
M = {'A': 34, 'B': 50, 'C': 50}

# Model
model = gp.Model('chair_ordering')

# Decision variables
x = model.addVars(manufacturers, vtype=GRB.INTEGER, lb=0, name="x")
y = model.addVars(manufacturers, vtype=GRB.BINARY, name="y")

# Linking constraints x[i] and y[i]
for i in manufacturers:
    model.addConstr(x[i] >= y[i], name=f"link_min_{i}")
    model.addConstr(x[i] <= M[i] * y[i], name=f"link_max_{i}")

# If A is used then B is used: y[A] <= y[B]
model.addConstr(y['A'] <= y['B'], name="A_implies_B")

# If B is used then C is used: y[B] <= y[C]
model.addConstr(y['B'] <= y['C'], name="B_implies_C")

# Total chairs constraints
model.addConstr(gp.quicksum(s[i] * x[i] for i in manufacturers) >= min_chairs,
                name="min_chairs")
model.addConstr(gp.quicksum(s[i] * x[i] for i in manufacturers) <= max_chairs,
                name="max_chairs")

# Objective: minimize total cost
model.setObjective(gp.quicksum(c[i] * x[i] for i in manufacturers), GRB.MINIMIZE)

# Optimize
model.optimize()

# Write optimal objective value to file
if model.status == GRB.OPTIMAL:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write(str(model.objVal))
else:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write('No optimal solution found')