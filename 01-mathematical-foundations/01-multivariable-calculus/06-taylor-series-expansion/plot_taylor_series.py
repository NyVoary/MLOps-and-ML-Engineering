"""
Visualizing the Taylor series approximation of f(x) = e^x around a = 0.

I plot:
    1. The true curve of e^x.
    2. The Taylor polynomials of order 0, 1, 2, and 3, all expanded
       around the same point a = 0 - each one hugs the true curve a
       little more closely near a.
    3. How the approximation error grows the further x gets from a:
       comparing x = 0.1 (very close to a) with x = 2.0 (far from a)
       at the same order N = 3.
"""

import numpy as np
import matplotlib.pyplot as plt

from taylor_series import taylor_expansion_1d


def f(x: np.ndarray) -> np.ndarray:
    """Function under study: f(x) = e^x."""
    return np.exp(x)


def taylor_polynomial(x_vals: np.ndarray, a: float, order_N: int) -> np.ndarray:
    """Evaluates the order-N Taylor polynomial of e^x around 'a' over an array of x values."""
    return np.array([taylor_expansion_1d(f, a, x, order_N)[0] for x in x_vals])


def plot(a: float = 0.0, save_path: str = "taylor_series_plot.png") -> None:
    x_vals = np.linspace(-2, 3, 300)
    true_curve = f(x_vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- 1. Successive orders hugging the true curve near a ---
    ax1 = axes[0]
    ax1.plot(x_vals, true_curve, color="black", linewidth=2, label="true f(x) = eˣ")
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    for order, color in zip(range(4), colors):
        approx = taylor_polynomial(x_vals, a, order)
        ax1.plot(x_vals, approx, color=color, linestyle="--", label=f"order {order}")
    ax1.scatter([a], [f(a)], color="black", zorder=5)
    ax1.set_ylim(-2, 20)
    ax1.set_title(f"Taylor polynomials of eˣ around a = {a}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend(fontsize=8)

    # --- 2. Error growing with distance from a, at fixed order N = 3 ---
    ax2 = axes[1]
    order_N = 3
    approx_3 = taylor_polynomial(x_vals, a, order_N)
    error = np.abs(true_curve - approx_3)
    ax2.plot(x_vals, error, color="purple")
    ax2.set_yscale("log")
    for x_mark, label in [(0.1, "x=0.1"), (2.0, "x=2.0")]:
        err_mark = abs(f(x_mark) - taylor_expansion_1d(f, a, x_mark, order_N)[0])
        ax2.scatter([x_mark], [err_mark], color="red", zorder=5)
        ax2.annotate(f"{label}\nerror={err_mark:.1e}", (x_mark, err_mark),
                     textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax2.set_title(f"Approximation error grows with |x - a| (order {order_N})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|true - approx|  (log scale)")

    fig.suptitle("A Taylor series is a *local* approximation: great near a, worse far away", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
