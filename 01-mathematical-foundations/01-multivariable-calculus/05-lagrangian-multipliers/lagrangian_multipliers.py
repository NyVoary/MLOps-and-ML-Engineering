import numpy as np
from scipy.optimize import minimize

# 1. Définition de la fonction objectif (on veut MAXIMISER f, donc MINIMISER -f)
def objective(v):
    x, y = v[0], v[1]
    return -(x + y)  # Signe moins car scipy sait seulement minimiser

# 2. Définition de la contrainte g(x, y) = x^2 + y^2 - 1 = 0
def constraint(v):
    x, y = v[0], v[1]
    return x**2 + y**2 - 1.0

if __name__ == "__main__":
    print("==================================================")
    print(" ⚖️ OPTIMISATION SOUS CONTRAINTE (LAGRANGE)")
    print("==================================================\n")

    # Point de départ initial (ex: x=0.1, y=0.1)
    x0 = [0.1, 0.1]

    # Structure de contrainte pour SciPy (type 'eq' pour égalité = 0)
    cons = ({'type': 'eq', 'fun': constraint})

    # Résolution
    res = minimize(objective, x0, constraints=cons)

    x_opt, y_opt = res.x[0], res.x[1]
    f_max = -res.fun  # On ré-inverse le signe pour retrouver le maximum

    print(f"Point optimal trouvé : (x = {x_opt:.4f}, y = {y_opt:.4f})")
    print(f"Valeur maximale f(x, y) : {f_max:.4f}")
    print(f"Valeur théorique (√2)   : {np.sqrt(2):.4f}")
    print(f"Contrainte respectée ?  : x² + y² = {x_opt**2 + y_opt**2:.4f} (Cible: 1.0)")
    print("==================================================")