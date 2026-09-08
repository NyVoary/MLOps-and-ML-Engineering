import numpy as np
import math

def numerical_derivative(f, x, n, h=1e-3):
    """
    Calcule la n-ième dérivée numérique d'une fonction f au point x
    en utilisant les différences finies (approximations successives).
    """
    if n == 0:
        return f(x)
    elif n == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif n == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    elif n == 3:
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * (h**3))
    else:
        # Pour n >= 4, approche récursive d'ordre 1
        return (numerical_derivative(f, x + h, n - 1, h) - 
                numerical_derivative(f, x - h, n - 1, h)) / (2 * h)

def taylor_expansion_1d(f, a, x, order_N):
    """
    Calcule l'approximation en Série de Taylor de f(x) autour du point 'a'
    jusqu'à l'ordre order_N de manière numérique.
    """
    approx = 0.0
    terms = []
    
    for n in range(order_N + 1):
        # 1. Calcul de la n-ième dérivée f^(n)(a)
        df_n = numerical_derivative(f, a, n)
        
        # 2. Calcul du terme (f^(n)(a) / n!) * (x - a)^n
        fact = math.factorial(n)
        term = (df_n / fact) * ((x - a)**n)
        
        approx += term
        terms.append(term)
        
    return approx, terms

# =====================================================================
# Démonstration et Validation
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" 📐 APPROXIMATION PAR SÉRIE DE TAYLOR (NUMÉRIQUE)")
    print("==================================================\n")

    # Fonction de test : exp(x) autour de a = 0
    f_test = np.exp
    a = 0.0
    x_eval = 0.1
    max_order = 3

    print(f"Fonction : f(x) = e^x")
    print(f"Point de développement (a) : {a}")
    print(f"Point d'évaluation (x)     : {x_eval}\n")

    true_val = f_test(x_eval)
    approx_val, terms = taylor_expansion_1d(f_test, a, x_eval, max_order)

    print("--- Résultats de l'approximation par ordre ---")
    cum_sum = 0
    for order, term in enumerate(terms):
        cum_sum += term
        print(f"Ordre {order} : terme = {term:.8f} | Cumul = {cum_sum:.8f}")

    print("\n--------------------------------------------------")
    print(f"Vraie valeur de e^({x_eval})  : {true_val:.8f}")
    print(f"Valeur estimée (Ordre {max_order}) : {approx_val:.8f}")
    print(f"Erreur absolue              : {abs(true_val - approx_val):.8e}")
    print("==================================================")