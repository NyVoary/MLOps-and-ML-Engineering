"""
Visualisation du vecteur gradient de f(x, y) = x^2 * y + 4 * y^2.

Je trace :
    1. Les lignes de niveau (contour plot) de f(x, y).
    2. Le champ de gradients (quiver) sur toute la grille : chaque flèche
       pointe vers la direction où f augmente le plus vite, localement.
    3. Le gradient au point d'évaluation, en évidence, perpendiculaire
       à la ligne de niveau qui passe par ce point — exactement la
       propriété géométrique du cours.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fonction d'étude : f(x, y) = x^2 * y + 4 * y^2."""
    return (x**2) * y + 4 * (y**2)


def analytical_gradient(x: float, y: float) -> np.ndarray:
    """Gradient exact : [df/dx, df/dy] = [2xy, x^2 + 8y]."""
    return np.array([2 * x * y, x**2 + 8 * y], dtype=float)


def plot(x0: float = 1.0, y0: float = 2.0, save_path: str = "gradient_vector_plot.png") -> None:
    grad = analytical_gradient(x0, y0)

    x_range = np.linspace(-3, 3, 200)
    y_range = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Grille plus grossière pour le champ de vecteurs (sinon illisible)
    xg = np.linspace(-3, 3, 12)
    yg = np.linspace(-3, 3, 12)
    XG, YG = np.meshgrid(xg, yg)
    U = 2 * XG * YG          # df/dx sur la grille
    V = XG**2 + 8 * YG       # df/dy sur la grille

    fig, ax = plt.subplots(figsize=(7, 6))
    contours = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=7)

    # Champ de gradients (normalisé pour la lisibilité)
    norm = np.sqrt(U**2 + V**2)
    norm[norm == 0] = 1
    ax.quiver(XG, YG, U / norm, V / norm, color="gray", alpha=0.6, scale=25)

    # Gradient au point d'évaluation, mis en évidence
    ax.scatter([x0], [y0], color="red", zorder=5, label=f"point ({x0}, {y0})")
    ax.quiver(
        x0, y0, grad[0], grad[1],
        color="red", scale=60, width=0.008,
        label=f"∇f({x0}, {y0}) = [{grad[0]:.1f}, {grad[1]:.1f}]",
    )

    ax.set_title("Lignes de niveau + champ de gradients de f(x, y) = x²y + 4y²")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"📊 Figure sauvegardée : {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
