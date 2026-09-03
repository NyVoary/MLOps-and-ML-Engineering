import numpy as np

# 1. Définition de la fonction d'étude : f(x, y) = x^2 * y + 4 * y^2
def f(x: float, y: float) -> float:
    return (x**2) * y + 4 * (y**2)

# 2. Application du pseudo-code pour la dérivée partielle par rapport à X
def partial_derivative_x(func, x: float, y: float, h: float = 1e-5) -> float:
    """
    Calcule la dérivée partielle ∂f/∂x au point (x, y) par différences finies centrées.
    Seul 'x' varie de +/- h, 'y' reste fixe.
    """
    f_plus = func(x + h, y)
    f_moins = func(x - h, y)
    
    df_dx = (f_plus - f_moins) / (2 * h)
    return df_dx

# 3. Application du pseudo-code pour la dérivée partielle par rapport à Y
def partial_derivative_y(func, x: float, y: float, h: float = 1e-5) -> float:
    """
    Calcule la dérivée partielle ∂f/∂y au point (x, y) par différences finies centrées.
    Seul 'y' varie de +/- h, 'x' reste fixe.
    """
    f_plus = func(x, y + h)
    f_moins = func(x, y - h)
    
    df_dy = (f_plus - f_moins) / (2 * h)
    return df_dy


if __name__ == "__main__":
    # Point de test
    x_val, y_val = 1.0, 2.0
    
    # Calculs numériques via nos fonctions
    df_dx_num = partial_derivative_x(f, x_val, y_val)
    df_dy_num = partial_derivative_y(f, x_val, y_val)
    
    # Valeurs théoriques exactes (Calculées à la main : ∂f/∂x = 2xy = 4, ∂f/∂y = x^2 + 8y = 17)
    df_dx_exact = 4.0
    df_dy_exact = 17.0
    
    print("==================================================")
    print(" 📐 CALCUL DES DÉRIVÉES PARTIELLES (NUMÉRIQUE)")
    print("==================================================")
    print(f"Point d'évaluation : (x = {x_val}, y = {y_val})\n")
    
    print(f"∂f/∂x calculé : {df_dx_num:.5f} | Valeur exacte : {df_dx_exact}")
    print(f"∂f/∂y calculé : {df_dy_num:.5f} | Valeur exacte : {df_dy_exact}\n")
    
    # Formation du vecteur gradient avec NumPy
    gradient_vector = np.array([df_dx_num, df_dy_num])
    print(f"🧭 Vecteur Gradient ∇f(1, 2) = {gradient_vector}")
    print("==================================================")