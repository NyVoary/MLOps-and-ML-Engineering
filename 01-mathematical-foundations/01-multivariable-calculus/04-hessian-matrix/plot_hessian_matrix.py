"""
Visualizing what the Hessian matrix tells me about curvature.

The Hessian's job is to tell apart a "bowl" (local minimum) from a
"mountain pass" (saddle point) even when the gradient is zero at both.
I show that side by side:

    1. f_bowl(x, y) = x^2 + y^2          -> Hessian is positive-definite
       everywhere (both eigenvalues > 0): a true minimum at (0, 0).
    2. f_saddle(x, y) = x^2 - y^2        -> Hessian has one positive and
       one negative eigenvalue: a saddle point at (0, 0) - going up along
       one axis, down along the other, exactly like a mountain pass.

Both functions have gradient = [0, 0] at the origin, so the gradient
alone can't distinguish them. Only the curvature (the Hessian) can.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for projection='3d')


def compute_hessian(f, point: np.ndarray, h: float = 1e-4) -> np.ndarray:
    point = np.array(point, dtype=float)
    n = len(point)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                p_plus, p_minus = np.copy(point), np.copy(point)
                p_plus[i] += h
                p_minus[i] -= h
                H[i, i] = (f(p_plus) - 2 * f(point) + f(p_minus)) / (h**2)
            else:
                p_pp, p_pm, p_mp, p_mm = (np.copy(point) for _ in range(4))
                p_pp[i] += h; p_pp[j] += h
                p_pm[i] += h; p_pm[j] -= h
                p_mp[i] -= h; p_mp[j] += h
                p_mm[i] -= h; p_mm[j] -= h
                H[i, j] = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4 * h**2)
    return H


def f_bowl(v: np.ndarray) -> float:
    x, y = v[0], v[1]
    return x**2 + y**2


def f_saddle(v: np.ndarray) -> float:
    x, y = v[0], v[1]
    return x**2 - y**2


def plot(save_path: str = "hessian_matrix_plot.png") -> None:
    x_range = np.linspace(-2, 2, 60)
    y_range = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(x_range, y_range)

    fig = plt.figure(figsize=(12, 5.5))

    for i, (name, func, cmap) in enumerate(
        [("Bowl: f = x² + y²  (minimum)", f_bowl, "viridis"),
         ("Saddle: f = x² - y²  (saddle point)", f_saddle, "coolwarm")]
    ):
        Z = np.array([[func(np.array([xi, yi])) for xi in x_range] for yi in y_range])
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0)

        H = compute_hessian(func, np.array([0.0, 0.0]))
        eigvals = np.linalg.eigvalsh(H)
        ax.scatter([0], [0], [func(np.array([0.0, 0.0]))], color="red", s=50)
        ax.set_title(f"{name}\nH eigenvalues: {np.round(eigvals, 1)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")

    fig.suptitle(
        "Same zero gradient at the origin - the Hessian's eigenvalues tell them apart",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
