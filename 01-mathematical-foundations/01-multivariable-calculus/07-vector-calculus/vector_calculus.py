import numpy as np

# =====================================================================
# 1. Opérateurs de Calcul Vectoriel (Numériques)
# =====================================================================

def compute_divergence_2d(F_x, F_y, x, y, h=1e-5):
    """
    Calcule la divergence d'un champ vectoriel F = (F_x, F_y) au point (x, y).
    div(F) = ∂F_x/∂x + ∂F_y/∂y
    """
    # Différences finies centrées pour dF_x / dx
    dFx_dx = (F_x(x + h, y) - F_x(x - h, y)) / (2 * h)
    
    # Différences finies centrées pour dF_y / dy
    dFy_dy = (F_y(x, y + h) - F_y(x, y - h)) / (2 * h)
    
    return dFx_dx + dFy_dy

def compute_laplacian_2d(f, x, y, h=1e-5):
    """
    Calcule le Laplacien d'une fonction scalaire f(x, y) au point (x, y).
    Δf = ∂²f/∂x² + ∂²f/∂y²
    """
    # Différences finies centrées de second ordre pour d²f / dx²
    d2f_dx2 = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h**2)
    
    # Différences finies centrées de second ordre pour d²f / dy²
    d2f_dy2 = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h**2)
    
    return d2f_dx2 + d2f_dy2


# =====================================================================
# 2. Démonstration & Validation
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" 🔄 CALCUL VECTORIEL : DIVERGENCE & LAPLACIEN")
    print("==================================================\n")

    # Point d'évaluation de test
    x_test, y_test = 2.0, 3.0

    # -----------------------------------------------------------------
    # Test 1 : Divergence du champ F(x, y) = [3x^2, 2y]
    # -----------------------------------------------------------------
    F_x = lambda x, y: 3 * x**2
    F_y = lambda x, y: 2 * y

    div_num = compute_divergence_2d(F_x, F_y, x_test, y_test)
    div_exact = 6 * x_test + 2  # Résultat exact calculé ensemble

    print("--- 🚰 DIVERGENCE ---")
    print(f"Champ Vectoriel : F(x, y) = [3x², 2y]")
    print(f"Point d'évaluation (x, y) : ({x_test}, {y_test})")
    print(f"Divergence Numérique       : {div_num:.6f}")
    print(f"Divergence Exacte (6x + 2) : {div_exact:.6f}")
    print(f"Erreur                     : {abs(div_num - div_exact):.2e}\n")

    # -----------------------------------------------------------------
    # Test 2 : Laplacien de f(x, y) = x^3 + y^3
    # -----------------------------------------------------------------
    f_scalar = lambda x, y: x**3 + y**3

    lap_num = compute_laplacian_2d(f_scalar, x_test, y_test)
    lap_exact = 6 * x_test + 6 * y_test  # Résultat exact calculé ensemble

    print("--- 🌐 LAPLACIEN ---")
    print(f"Fonction Scalaire : f(x, y) = x³ + y³")
    print(f"Point d'évaluation (x, y) : ({x_test}, {y_test})")
    print(f"Laplacien Numérique          : {lap_num:.6f}")
    print(f"Laplacien Exact (6x + 6y)    : {lap_exact:.6f}")
    print(f"Erreur                       : {abs(lap_num - lap_exact):.2e}")
    print("==================================================")