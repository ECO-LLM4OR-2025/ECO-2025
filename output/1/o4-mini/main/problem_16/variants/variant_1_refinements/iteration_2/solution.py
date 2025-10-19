import gurobipy as gp
from gurobipy import GRB

# Data
T = [1, 2, 3]  # months
p_buy  = {1: 8,  2: 6,  3: 9}   # unit purchase prices
p_sell = {1: 9,  2: 8,  3: 10}  # unit selling prices
I0     = 200                   # initial inventory at start of month 1
C      = 500                   # warehouse capacity (units)
# Monthly demand forecasts (realistic limits on how much can be sold each month)
demand = {1: 200, 2: 250, 3: 300}
h      = 0.5   # holding cost per unit carried from one month to the next
salvage_value = 0.0  # salvage value per unit of inventory left at end of month 3

try:
    # Create model
    m = gp.Model("quarterly_inventory")
    m.ModelSense = GRB.MAXIMIZE
    m.setParam('OutputFlag', 0)  # silent solve

    # Decision variables (integer units)
    q = m.addVars(T, vtype=GRB.INTEGER, name="q")  # purchased units in month t
    s = m.addVars(T, vtype=GRB.INTEGER, name="s")  # sold units in month t
    I = m.addVars(T, lb=0, vtype=GRB.INTEGER, name="I")  # end-of-month inventory

    # Inventory balance constraints
    for t in T:
        if t == 1:
            # Beginning inventory I0 + purchases - sales = end-of-month inventory
            m.addConstr(I0 + q[t] - s[t] == I[t], name="balance_1")
        else:
            m.addConstr(I[t-1] + q[t] - s[t] == I[t], name=f"balance_{t}")

    # Warehouse capacity
    m.addConstrs((I[t] <= C for t in T), name="capacity")

    # Demand constraints (cannot sell more than forecast demand)
    m.addConstrs((s[t] <= demand[t] for t in T), name="demand_limit")

    # Objective: maximize profit = sales revenue - purchase cost 
    #                          - holding cost on I1 & I2 + salvage on I3
    revenue = gp.quicksum(p_sell[t] * s[t] for t in T)
    purchase_cost = gp.quicksum(p_buy[t] * q[t] for t in T)
    holding_cost  = h * (I[1] + I[2])        # no holding cost on final inventory
    salvage_gain  = salvage_value * I[3]     # salvage value for leftover in month 3

    m.setObjective(revenue - purchase_cost - holding_cost + salvage_gain)

    # Solve
    m.optimize()

    # Write out the optimal value (numeric only) to file
    with open('ref_optimal_value.txt', 'w') as f:
        if m.status == GRB.OPTIMAL:
            f.write(str(m.objVal))
        else:
            # no feasible/optimal solution found
            f.write("")

except gp.GurobiError as e:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("")
    print("Gurobi error:", e)

except Exception as ex:
    with open('ref_optimal_value.txt', 'w') as f:
        f.write("")
    print("Unexpected error:", ex)