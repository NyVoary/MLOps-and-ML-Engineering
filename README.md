# MLOps & ML Engineering — Learning Journal 📓

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-numerical-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SymPy](https://img.shields.io/badge/SymPy-symbolic-3B5526?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-planned-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-planned-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-planned-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-planned-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Roadmap](https://img.shields.io/badge/roadmap.sh-ML%20Engineering%20%26%20MLOps-2F80ED?logo=roadmapdotsh&logoColor=white)](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops)

This repository is my **personal learning journal** as I follow the [Machine Learning Engineering and MLOps](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops) roadmap on [roadmap.sh](https://roadmap.sh/).

The goal isn't to build a library or a finished product, but to **keep a written, version-controlled record** of what I learn: theory, notes, exercises, scripts, and small "from scratch" implementations before moving on to the standard tools/frameworks of the field.

## 🗂️ Repository organization

The roadmap has three levels — **Section** → **Sub-section** → **Topic** (e.g. *Mathematical Foundations* → *Multivariable Calculus* → *Jacobian Matrices*). The folder structure only mirrors the first two levels; individual topics live **as sections inside the sub-section's `README.md`**, not as their own subfolders:

- Each **major roadmap section** becomes a **numbered folder** at the root (e.g. `01-mathematical-foundations`).
- Each **roadmap sub-section** becomes a **numbered subfolder** inside it (e.g. `01-multivariable-calculus`).
- Each **individual topic** of that sub-section (e.g. Partial Derivatives, Gradient Vectors, Jacobian Matrices, Hessian Matrices...) becomes a **new section in the same `README.md`**, plus its own script(s) when it warrants code.

```
01-mathematical-foundations/
└── 01-multivariable-calculus/            # one folder per roadmap sub-section
    ├── README.md                          # all topics of this sub-section, one section each:
    │                                       #   1. Partial Derivatives
    │                                       #   2. Gradient Vectors
    │                                       #   3. Jacobian Matrices        (added later)
    │                                       #   4. Hessian Matrices         (added later)
    │                                       #   ...
    ├── partial_derivatives.py             # "from scratch" implementation
    ├── partial_derivatives_and_gradients.py
    ├── interactive_partial_derivatives.py # comparison with a standard library (SymPy/NumPy)
    └── jacobian_matrices.py               # (added later, alongside its README section)
02-<next-section>/
└── 01-<first-sub-section>/
    └── README.md
...
```

This keeps one folder per sub-section instead of one per leaf topic — less nesting, and topics that are naturally studied together (like Partial Derivatives and Gradient Vectors) stay next to each other.

## 📄 What goes in each subfolder?

**1. A summary sheet (`README.md`)**
A short, synthetic sheet summarizing the mathematical or algorithmic intuition behind each topic of the sub-section: definitions, notation, a worked example, pseudo-code. The goal is to be able to re-read it later and understand the concept without re-deriving everything.

**2. Implementation scripts (`.py` or `.ipynb`)**

- **Code the concept from scratch** (e.g. rewrite gradient descent or derivative computation with NumPy), to really understand the mechanism under the hood.
- **Use the industry-standard library** (e.g. SciPy, PyTorch, or Scikit-Learn) to compare results and validate the "from scratch" implementation.

## ✅ Progress tracker

- [ ] **01. Mathematical Foundations**
  - [ ] 01.1 [Multivariable Calculus](01-mathematical-foundations/01-multivariable-calculus/README.md)
    - [x] Partial Derivatives
    - [x] Gradient Vectors

*(Check off items as you progress through the roadmap; each sub-section links to its folder's README, where individual topics get their own section. New topics/sub-sections are added here as they're covered.)*

## 🛠️ Code conventions

- **Type hints** on every function.
- **Docstrings** with examples.
- **Doctests** to validate examples directly from the docstring.
- When relevant: comparing an **analytical/exact** approach vs. a **numerical** approach (finite differences) to validate the implementation.

## 🔗 Resources

- Roadmap followed: [Machine Learning Engineering and MLOps](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops)
