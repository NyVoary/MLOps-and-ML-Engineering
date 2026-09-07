import numpy as np

def compute_hessian(f, point, h=1e-4):
    """
    Calcule numériquement la matrice Hessienne d'une fonction scalaire f
    au point d'évaluation 'point' en utilisant la méthode des différences finies centrées.

    Paramètres:
        f     : fonction scalaire qui prend un vecteur NumPy (taille n) et renvoie un réel
        point : vecteur d'entrée NumPy ou liste [x1, x2, ..., xn]
        h     : pas de variation pour les différences finies (1e-4 est idéal pour f'')

    Retourne:
        H     : matrice Hessienne NumPy de forme (n, n)
    """
    point = np.array(point, dtype=float)
    n = len(point)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # 1. Dérivée seconde pure : d^2f / dx_i^2
                # Formule : (f(x + h) - 2*f(x) + f(x - h)) / h^2
                point_plus = np.copy(point)
                point_minus = np.copy(point)
                
                point_plus[i] += h
                point_minus[i] -= h
                
                f_plus = f(point_plus)
                f_ref = f(point)
                f_minus = f(point_minus)
                
                H[i, i] = (f_plus - 2 * f_ref + f_minus) / (h ** 2)
            else:
                # 2. Dérivée seconde croisée : d^2f / (dx_i dx_j)
                # Formule à 4 points pour la différence finie centrée croisée
                p_pp = np.copy(point) # +h_i, +h_j
                p_pm = np.copy(point) # +h_i, -h_j
                p_mp = np.copy(point) # -h_i, +h_j
                p_mm = np.copy(point) # -h_i, -h_j
                
                p_pp[i] += h; p_pp[j] += h
                p_pm[i] += h; p_pm[j] -= h
                p_mp[i] -= h; p_mp[j] += h
                p_mm[i] -= h; p_mm[j] -= h
                
                H[i, j] = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4 * (h ** 2))
                
    return H

# =====================================================================
# Démonstration et Validation
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" 📐 CALCULATEUR DE MATRICE HESSIENNE (NUMPY)")
    print("==================================================\n")

    # Notre fonction test : f(x, y) = 3*(x^2)*y + 2*(y^3) - 5*x
    def loss_function(v):
        x, y = v[0], v[1]
        return 3 * (x**2) * y + 2 * (y**3) - 5 * x

    pt = [1.0, 2.0]
    H_calc = compute_hessian(loss_function, pt)

    print("--- EXEMPLE : f(x, y) = 3x^2y + 2y^3 - 5x ---")
    print(f"Point d'évaluation : (x={pt[0]}, y={pt[1]})")
    print(f"Forme de la matrice Hessienne : {H_calc.shape}")
    print("\nMatrice Hessienne calculée H(1, 2) :")
    print(np.round(H_calc, 4))
    
    print("\nMatrice Théorique attendue :")
    print("[[12.  6.]\n [ 6. 24.]]")
    
    # Vérification de la symétrie (Théorème de Schwarz)
    is_symmetric = np.allclose(H_calc, H_calc.T)
    print(f"\nEst-ce que la matrice est symétrique ? {is_symmetric} 🪞")
    print("==================================================")