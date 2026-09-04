import numpy as np

def compute_jacobian(F, point, h=1e-5):
    """
    Calcule numériquement la matrice Jacobienne d'une fonction vectorielle F
    au point d'évaluation 'point' en utilisant la méthode des différences finies centrées.

    Paramètres:
        F     : fonction qui prend un vecteur NumPy (taille n) et renvoie un vecteur NumPy (taille m)
        point : vecteur d'entrée NumPy ou liste [x1, x2, ..., xn]
        h     : pas de variation pour les différences finies (par défaut 1e-5)

    Retourne:
        J     : matrice Jacobienne NumPy de forme (m, n)
    """
    point = np.array(point, dtype=float)
    
    # 1. Évaluation de référence pour connaître la taille de la sortie (m)
    F_ref = F(point)
    m = len(F_ref)
    n = len(point)
    
    # 2. Initialisation de la matrice Jacobienne (m x n) remplie de zéros
    J = np.zeros((m, n))
    
    # 3. Calcul de chaque colonne j de la Jacobienne (sensibilité par rapport à la variable x_j)
    for j in range(n):
        point_plus = np.copy(point)
        point_minus = np.copy(point)
        
        # Perturbation de la j-ème variable
        point_plus[j] += h
        point_minus[j] -= h
        
        # Calcul des sorties perturbées
        F_plus = F(point_plus)
        F_minus = F(point_minus)
        
        # Différence finie centrée pour la colonne j : (F(x+h) - F(x-h)) / (2h)
        J[:, j] = (F_plus - F_minus) / (2 * h)
        
    return J

# =====================================================================
# Exemples de démonstration
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" 📐 CALCULATEUR DE MATRICE JACOBIENNE (NUMPY)")
    print("==================================================\n")

    # --- Exemple 1 : Fonction R^2 -> R^2 (Notre exemple calculé à la main) ---
    # F(x, y) = [x^2 * y, x + 3y]
    def example_function(v):
        x, y = v[0], v[1]
        f1 = (x**2) * y
        f2 = x + 3 * y
        return np.array([f1, f2])

    point_1 = [2.0, 3.0]
    J1 = compute_jacobian(example_function, point_1)

    print("--- EXEMPLE 1 : F(x, y) = [x^2 * y, x + 3y] ---")
    print(f"Point d'évaluation : {point_1}")
    print(f"Forme de la matrice Jacobienne (m x n) : {J1.shape}")
    print("Matrice Jacobienne J(2, 3) :")
    print(J1)
    print("Calcul théorique attendu  : [[12.  4.]")
    print("                            [ 1.  3.]]\n")

    # --- Exemple 2 : Couche de Réseau de Neurones R^3 -> R^2 ---
    # Z = W * X + B (2 neurones de sortie, 3 variables d'entrée)
    W = np.array([
        [0.5, -1.2,  2.0],  # Poids du Neurone 1
        [1.5,  0.8, -0.4]   # Poids du Neurone 2
    ])
    b = np.array([0.1, -0.5])

    def neural_layer(X):
        return np.dot(W, X) + b

    X_in = [1.0, 2.0, -1.0]
    J2 = compute_jacobian(neural_layer, X_in)

    print("--- EXEMPLE 2 : Couche Dense R^3 -> R^2 (Neural Network) ---")
    print(f"Vecteur d'entrée X : {X_in}")
    print(f"Forme de la matrice Jacobienne : {J2.shape} (2 sorties, 3 entrées)")
    print("Matrice Jacobienne dZ/dX :")
    print(J2)
    print("\nMatrice de Poids W initiale :")
    print(W)
    print("-> Remarque : La Jacobienne d'une couche linéaire correspond exactement aux Poids W !")
    print("==================================================")