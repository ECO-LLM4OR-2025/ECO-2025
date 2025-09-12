import gurobipy as gp
from gurobipy import GRB

# Data
E = [1, 3, 5]
L = [10, 12, 15]
T = [4, 8, 14]
alpha = [10, 20, 30]
beta = [5, 10, 15]
sep_matrix = [
    [0, 2, 3],
    [2, 0, 4],
    [3, 4, 0]
]

n = len(E)
I = range(n)

# Create model
model = gp.Model()

# Decision variables
# x[i]: landing time, bounded by earliest E[i] and latest L[i]
# eplus[i], eminus[i]: tardiness and earliness
x = model.addVars(I, lb=E, ub=L, name="x")
eplus = model.addVars(I, lb=0.0, name="eplus")
eminus = model.addVars(I, lb=0.0, name="eminus")

# Earliness/tardiness definition
for i in I:
    model.addConstr(
        x[i] - T[i] == eplus[i] - eminus[i],
        name=f"pd_{i}"
    )

# Separation constraints (fixed order: i < j)
for i in I:
    for j in I:
        if i < j:
            model.addConstr(
                x[j] >= x[i] + sep_matrix[i][j],
                name=f"sep_{i}_{j}"
            )

# Objective: minimize total earliness & tardiness penalties
model.setObjective(
    gp.quicksum(alpha[i] * eplus[i] + beta[i] * eminus[i] for i in I),
    GRB.MINIMIZE
)

# Solve
model.optimize()

# Save optimal value
opt_val = model.ObjVal
with open('ref_optimal_value.txt', 'w') as f:
    f.write(str(opt_val))