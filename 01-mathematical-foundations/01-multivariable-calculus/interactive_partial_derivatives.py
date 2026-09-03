import sympy as sp
import numpy as np

# 1. SymPy gère les symboles et l'affichage
x, y = sp.symbols('x y')
f_expr = x**2 * y + 4 * y**2

# Formules symboliques pour l'affichage à l'écran
df_dx_expr = sp.diff(f_expr, x)
df_dy_expr = sp.diff(f_expr, y)

print(f"📝 Affichage SymPy : ∂f/∂x = {df_dx_expr}")

# 2. Conversion en fonction Python/NumPy ultra-rapide
f_num = sp.lambdify((x, y), f_expr, 'numpy')

# 3. NumPy effectue le calcul numérique par différences finies au point (1.0, 2.0)
x_val, y_val = 1.0, 2.0
h = 1e-5

df_dx_numpy = (f_num(x_val + h, y_val) - f_num(x_val - h, y_val)) / (2 * h)
df_dy_numpy = (f_num(x_val, y_val + h) - f_num(x_val, y_val - h)) / (2 * h)

gradient_vector = np.array([df_dx_numpy, df_dy_numpy])
print(f"🧭 Gradient évalué via NumPy : {gradient_vector}")