"""
Visualizing the gradient vector of f(x, y) = x^2 * y + 4 * y^2.

I plot:
    1. The level curves (contour plot) of f(x, y).
    2. The gradient field (quiver) over the whole grid: every arrow points
       toward the direction where f increases the fastest, locally.
    3. The gradient at the evaluation point, highlighted, perpendicular
       to the level curve passing through it — exactly the geometric
       property from the theory.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Function under study: f(x, y) = x^2 * y + 4 * y^2."""
    return (x**2) * y + 4 * (y**2)


def analytical_gradient(x: float, y: float) -> np.ndarray:
    """Exact gradient: [df/dx, df/dy] = [2xy, x^2 + 8y]."""
    return np.array([2 * x * y, x**2 + 8 * y], dtype=float)


def plot(x0: float = 1.0, y0: float = 2.0, save_path: str = "gradient_vector_plot.png") -> None:
    grad = analytical_gradient(x0, y0)

    x_range = np.linspace(-3, 3, 200)
    y_range = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Coarser grid for the vector field (otherwise unreadable)
    xg = np.linspace(-3, 3, 12)
    yg = np.linspace(-3, 3, 12)
    XG, YG = np.meshgrid(xg, yg)
    U = 2 * XG * YG          # df/dx over the grid
    V = XG**2 + 8 * YG       # df/dy over the grid

    fig, ax = plt.subplots(figsize=(7, 6))
    contours = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=7)

    # Gradient field (normalized for readability)
    norm = np.sqrt(U**2 + V**2)
    norm[norm == 0] = 1
    ax.quiver(XG, YG, U / norm, V / norm, color="gray", alpha=0.6, scale=25)

    # Gradient at the evaluation point, highlighted
    ax.scatter([x0], [y0], color="red", zorder=5, label=f"point ({x0}, {y0})")
    ax.quiver(
        x0, y0, grad[0], grad[1],
        color="red", scale=60, width=0.008,
        label=f"∇f({x0}, {y0}) = [{grad[0]:.1f}, {grad[1]:.1f}]",
    )

    ax.set_title("Level curves + gradient field of f(x, y) = x²y + 4y²")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
