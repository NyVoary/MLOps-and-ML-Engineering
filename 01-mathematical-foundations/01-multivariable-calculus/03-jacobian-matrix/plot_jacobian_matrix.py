"""
Visualizing the Jacobian matrix of F(x, y) = [x^2 * y, x + 3y].

Since F maps R^2 -> R^2, I can't draw one surface like for a scalar
function. Instead, I visualize what the Jacobian *means* geometrically:
how F deforms space.

I plot:
    1. A grid of points in the input space (x, y).
    2. The same grid after being passed through F — showing how the
       straight lines get bent into curves (F is nonlinear).
    3. The linear approximation of F near the evaluation point, using
       only the Jacobian: F(x0+dx, y0+dy) ≈ F(x0, y0) + J @ [dx, dy].
       This approximation lines up with the true curved grid right at
       the evaluation point, then drifts away — exactly what "local
       linear approximation" means.
"""

import numpy as np
import matplotlib.pyplot as plt


def F(point: np.ndarray) -> np.ndarray:
    """Vector-valued function under study: F(x, y) = [x^2 * y, x + 3y]."""
    x, y = point[0], point[1]
    return np.array([(x**2) * y, x + 3 * y])


def compute_jacobian(func, point: np.ndarray, h: float = 1e-5) -> np.ndarray:
    point = np.array(point, dtype=float)
    n = len(point)
    f_ref = func(point)
    m = len(f_ref)
    J = np.zeros((m, n))
    for j in range(n):
        point_plus = np.copy(point)
        point_minus = np.copy(point)
        point_plus[j] += h
        point_minus[j] -= h
        J[:, j] = (func(point_plus) - func(point_minus)) / (2 * h)
    return J


def plot(x0: float = 2.0, y0: float = 3.0, save_path: str = "jacobian_matrix_plot.png") -> None:
    point0 = np.array([x0, y0])
    J = compute_jacobian(F, point0)
    F0 = F(point0)

    # A small grid of lines around the evaluation point, in input space
    span = 1.0
    lines = np.linspace(-span, span, 7)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- 1. Input space grid ---
    ax_in = axes[0]
    for c in lines:  # vertical-ish lines: x = x0 + c
        xs = np.full(50, x0 + c)
        ys = np.linspace(y0 - span, y0 + span, 50)
        ax_in.plot(xs, ys, color="steelblue", linewidth=1)
    for c in lines:  # horizontal-ish lines: y = y0 + c
        xs = np.linspace(x0 - span, x0 + span, 50)
        ys = np.full(50, y0 + c)
        ax_in.plot(xs, ys, color="orange", linewidth=1)
    ax_in.scatter([x0], [y0], color="red", zorder=5, label=f"({x0}, {y0})")
    ax_in.set_title("Input space (x, y)")
    ax_in.set_xlabel("x")
    ax_in.set_ylabel("y")
    ax_in.legend()
    ax_in.set_aspect("equal")

    # --- 2. Output space: true F(grid) vs. Jacobian's linear approximation ---
    ax_out = axes[1]
    for c in lines:
        xs = np.full(50, x0 + c)
        ys = np.linspace(y0 - span, y0 + span, 50)
        pts = np.stack([F(np.array([xi, yi])) for xi, yi in zip(xs, ys)])
        ax_out.plot(pts[:, 0], pts[:, 1], color="steelblue", linewidth=1)
    for c in lines:
        xs = np.linspace(x0 - span, x0 + span, 50)
        ys = np.full(50, y0 + c)
        pts = np.stack([F(np.array([xi, yi])) for xi, yi in zip(xs, ys)])
        ax_out.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=1)

    # Jacobian's straight-line (linear) approximation, for comparison
    dx = np.linspace(-span, span, 2)
    for c in lines:
        d = np.stack([[dxi, c] for dxi in dx])
        approx = F0 + d @ J.T
        ax_out.plot(approx[:, 0], approx[:, 1], "k--", linewidth=0.6, alpha=0.6)
    for c in lines:
        d = np.stack([[c, dyi] for dyi in dx])
        approx = F0 + d @ J.T
        ax_out.plot(approx[:, 0], approx[:, 1], "k--", linewidth=0.6, alpha=0.6)

    ax_out.scatter([F0[0]], [F0[1]], color="red", zorder=5, label=f"F({x0}, {y0}) = ({F0[0]:.1f}, {F0[1]:.1f})")
    ax_out.plot([], [], "k--", label="linear approx. via J (dashed)")
    ax_out.set_title("Output space F(x, y) — curved grid vs. Jacobian's straight-line approximation")
    ax_out.set_xlabel("f1")
    ax_out.set_ylabel("f2")
    ax_out.legend(fontsize=8)

    fig.suptitle("The Jacobian is the best *local linear* approximation of a nonlinear map", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
