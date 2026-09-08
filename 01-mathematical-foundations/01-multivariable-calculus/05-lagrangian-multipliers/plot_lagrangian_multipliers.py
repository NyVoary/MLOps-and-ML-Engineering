"""
Visualizing Lagrange multipliers on maximize f(x, y) = x + y subject to
g(x, y) = x^2 + y^2 = 1.

I plot:
    1. The constraint g(x, y) = 1 as a circle.
    2. Several level curves of f(x, y) = x + y (diagonal lines).
    3. The optimal point (1/sqrt(2), 1/sqrt(2)), where the constraint
       circle is exactly tangent to a level curve of f.
    4. Both gradients, grad(f) and grad(g), at that point - they point
       in the same direction (parallel), which is the geometric meaning
       of grad(f) = lambda * grad(g).
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Objective function: f(x, y) = x + y."""
    return x + y


def grad_f(x: float, y: float) -> np.ndarray:
    return np.array([1.0, 1.0])


def grad_g(x: float, y: float) -> np.ndarray:
    """Gradient of the constraint g(x, y) = x^2 + y^2."""
    return np.array([2 * x, 2 * y])


def plot(save_path: str = "lagrangian_multipliers_plot.png") -> None:
    # Analytical optimum: x = y = 1/sqrt(2), lambda = 1/sqrt(2)
    x_opt = y_opt = 1 / np.sqrt(2)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x, circle_y = np.cos(theta), np.sin(theta)

    x_range = np.linspace(-1.8, 1.8, 300)
    y_range = np.linspace(-1.8, 1.8, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Level curves of f (diagonal lines x + y = c)
    levels = np.linspace(-2, 2, 17)
    contours = ax.contour(X, Y, Z, levels=levels, cmap="Greens", alpha=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="f=%.1f")

    # Constraint circle g(x, y) = 1
    ax.plot(circle_x, circle_y, color="steelblue", linewidth=2, label="constraint g(x, y) = 1")

    # Optimal point
    ax.scatter([x_opt], [y_opt], color="red", zorder=5, label=f"optimum ({x_opt:.2f}, {y_opt:.2f})")

    # Gradients at the optimum, scaled down for readability
    gf = grad_f(x_opt, y_opt)
    gg = grad_g(x_opt, y_opt)
    ax.quiver(x_opt, y_opt, *gf, color="darkgreen", scale=8, width=0.012, label="∇f (objective)")
    ax.quiver(x_opt, y_opt, *gg, color="darkorange", scale=16, width=0.012, label="∇g (constraint)")

    ax.set_title("At the optimum, ∇f and ∇g are parallel: ∇f = λ·∇g")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
