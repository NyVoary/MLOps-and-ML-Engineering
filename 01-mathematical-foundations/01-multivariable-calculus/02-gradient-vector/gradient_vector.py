import numpy as np

def f(x: float, y: float) -> float:
    """
    Fonction scalaire f(x, y) = x^2 * y + 4 * y^2
    """
    return (x**2) * y + 4 * (y**2)


def analytical_gradient(x: float, y: float) -> np.ndarray:
    """
    Calcule le gradient exact (analytique) de f(x, y).
    
    Grad f(x, y) = [ df/dx, df/dy ]
                 = [ 2*x*y, x^2 + 8*y ]
    """
    df_dx = 2 * x * y
    df_dy = x**2 + 8 * y
    return np.array([df_dx, df_dy], dtype=float)


def numerical_gradient(func, point: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Calcule le gradient numérique d'une fonction f par différences finies centrées.
    
    Args:
        func: La fonction scalaire à dériver (prend un tableau de coordonnées).
        point: Le point [x, y] où évaluer le gradient.
        h: Le pas de variation infinitésimal.
        
    Returns:
        np.ndarray: Le vecteur gradient approximé [df/dx, df/dy].
    """
    grad = np.zeros_like(point, dtype=float)
    
    for i in range(len(point)):
        # Point perturbé dans la direction +h
        point_plus = np.array(point, dtype=float)
        point_plus[i] += h
        
        # Point perturbé dans la direction -h
        point_minus = np.array(point, dtype=float)
        point_minus[i] -= h
        
        # Formule des différences finies centrées
        grad[i] = (func(*point_plus) - func(*point_minus)) / (2 * h)
        
    return grad


if __name__ == "__main__":
    # Point de test
    x_test, y_test = 1.0, 2.0
    point_test = np.array([x_test, y_test])

    print("==================================================")
    print(" 📐 CALCUL DE DÉRIVÉES PARTIELLES & GRADIENTS")
    print("==================================================")
    print(f"Fonction d'étude : f(x, y) = x^2 * y + 4 * y^2")
    print(f"Point d'évaluation : (x = {x_test}, y = {y_test})\n")

    # 1. Calcul Analytique (Exact)
    grad_exact = analytical_gradient(x_test, y_test)
    print(f"1. Gradient Analytique (Exact) :")
    print(f"   df/dx = {grad_exact[0]}")
    print(f"   df/dy = {grad_exact[1]}")
    print(f"   Vecteur grad(f) = {grad_exact}\n")

    # 2. Calcul Numérique (Différences Finies)
    grad_num = numerical_gradient(f, point_test)
    print(f"2. Gradient Numérique (Approximé avec h=1e-5) :")
    print(f"   Vecteur grad(f) = {grad_num}\n")

    # 3. Vérification de la précision (Erreur relative)
    error = np.linalg.norm(grad_exact - grad_num)
    print(f"3. Validation :")
    print(f"   Erreur absolue (Norme L2) = {error:.8f}")
    
    if np.allclose(grad_exact, grad_num, atol=1e-4):
        print("   ✅ Succès : La méthode numérique valide le calcul analytique !")
    else:
        print("   ❌ Erreur : Écart trop important entre les résultats.")
    print("==================================================")