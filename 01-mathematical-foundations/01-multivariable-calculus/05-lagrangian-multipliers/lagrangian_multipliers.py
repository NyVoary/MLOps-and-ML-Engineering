import numpy as np
from scipy.optimize import minimize

# 1. Objective function (I want to MAXIMIZE f, so I MINIMIZE -f instead)
def objective(v):
    x, y = v[0], v[1]
    return -(x + y)  # minus sign because SciPy only knows how to minimize

# 2. Constraint g(x, y) = x^2 + y^2 - 1 = 0
def constraint(v):
    x, y = v[0], v[1]
    return x**2 + y**2 - 1.0

if __name__ == "__main__":
    print("==================================================")
    print(" CONSTRAINED OPTIMIZATION (LAGRANGE)")
    print("==================================================\n")

    # Initial starting point (e.g. x=0.1, y=0.1)
    x0 = [0.1, 0.1]

    # Constraint structure for SciPy ('eq' type for an equality constraint = 0)
    cons = ({'type': 'eq', 'fun': constraint})

    # Solve
    res = minimize(objective, x0, constraints=cons)

    x_opt, y_opt = res.x[0], res.x[1]
    f_max = -res.fun  # flip the sign back to get the actual maximum

    print(f"Optimal point found: (x = {x_opt:.4f}, y = {y_opt:.4f})")
    print(f"Maximum value f(x, y): {f_max:.4f}")
    print(f"Theoretical value (sqrt(2)): {np.sqrt(2):.4f}")
    print(f"Constraint respected?: x^2 + y^2 = {x_opt**2 + y_opt**2:.4f} (target: 1.0)")
    print("==================================================")
