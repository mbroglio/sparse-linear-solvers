# Iterative Solvers for Large Sparse Linear Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Sparse-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Numba](https://img.shields.io/badge/JIT-Numba_Accelerated-00A3E0.svg)](https://numba.pydata.org/)
[![Academic Report](https://img.shields.io/badge/Report-Assignment1_PDF-red.svg)](Assignment1_Report.pdf)

> High-performance implementation, mathematical convergence analysis, and Numba JIT acceleration of iterative algorithms (**Jacobi**, **Gauss-Seidel**, **Gradient Descent**, **Conjugate Gradient**) for solving large sparse linear systems $Ax = b$.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Mathematical Foundation & Solvers](#-mathematical-foundation--solvers)
  - [Stationary Iterative Methods (Jacobi & Gauss-Seidel)](#1-stationary-iterative-methods)
  - [Krylov Subspace & Optimization Methods (Gradient & Conjugate Gradient)](#2-krylov-subspace--optimization-methods)
- [Numba JIT Acceleration](#-numba-jit-acceleration)
- [Benchmark Sparse Matrices](#-benchmark-sparse-matrices)
- [Execution & Experiments](#-execution--experiments)
- [Repository Structure](#-repository-structure)
- [Authors & Academic Context](#-authors--academic-context)

---

## 🔬 Project Overview

Solving large, sparse linear systems of equations of the form:
$$A x = b, \quad A \in \mathbb{R}^{n \times n}, \quad b \in \mathbb{R}^n$$
is a foundational problem in numerical computing, physical simulation, differential equation discretizations (FEM/VEM), and scientific optimization. Direct solvers (e.g. LU or Cholesky factorizations) suffer from fill-in that severely limits scalability when $n$ is large.

This project investigates iterative solvers across two implementations:
1. **Baseline Implementation**: Vectorized routines built on SciPy Compressed Sparse Row (`csr_matrix`) structures.
2. **Numba JIT-Accelerated Implementation**: Low-level, cache-friendly C-speed loops compiled via `@njit` leveraging multi-threaded sparse matrix-vector kernels.

---

## 📐 Mathematical Foundation & Solvers

### 1. Stationary Iterative Methods

Let $A = D - L - U$ be the standard matrix splitting into diagonal ($D$), strictly lower triangular ($L$), and strictly upper triangular ($U$) components.

#### Jacobi Method
Updates each unknown using only values from the previous iteration:
$$x^{(k+1)} = D^{-1} \left( (L + U) x^{(k)} + b \right)$$
- Highly parallelizable since component updates are decoupled.
- Converges if $A$ is strictly diagonally dominant.

#### Gauss-Seidel Method
Immediately uses updated components within the current iteration:
$$x^{(k+1)} = (D - L)^{-1} \left( U x^{(k)} + b \right)$$
- Typically exhibits twice the asymptotic convergence rate of Jacobi for positive-definite systems.

---

### 2. Krylov Subspace & Optimization Methods

Applicable when $A$ is Symmetric Positive Definite (SPD). Solving $Ax = b$ is equivalent to minimizing the quadratic functional:
$$\phi(x) = \frac{1}{2} x^T A x - b^T x$$

#### Gradient Descent (Steepest Descent)
Steps along the negative gradient (residual $r_k = b - A x_k$):
$$\alpha_k = \frac{r_k^T r_k}{r_k^T A r_k}, \qquad x_{k+1} = x_k + \alpha_k r_k$$

#### Conjugate Gradient (CG)
Searches along $A$-orthogonal (conjugate) directions $p_k$:
$$\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}, \quad x_{k+1} = x_k + \alpha_k p_k, \quad r_{k+1} = r_k - \alpha_k A p_k, \quad \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}, \quad p_{k+1} = r_{k+1} + \beta_k p_k$$
- Theoretical convergence in at most $n$ iterations (in exact arithmetic).
- Highly effective for ill-conditioned sparse systems when combined with preconditioning.

---

## ⚡ Numba JIT Acceleration

Iterative solvers with forward/backward triangular sweeps (like Gauss-Seidel) cannot be easily vectorized with pure NumPy because of sequential loop dependencies. 

The `linear_solvers/numba/` implementation compiles explicit loops over the CSR indices (`row_ptr`, `col_indices`, `data`) directly to native machine instructions via LLVM, delivering speedups of **10x to 50x** over pure Python implementations.

---

## 📊 Benchmark Sparse Matrices

The repository includes real-world and synthetic sparse benchmark matrices stored in Matrix Market format (`.mtx` in `data/`):

| Matrix | Origin / Application | Dimensions ($n \times n$) | Non-Zeros ($nnz$) | Properties |
| :--- | :--- | :---: | :---: | :--- |
| `vem1.mtx` | Virtual Element Method | Discretization mesh 1 | Sparse | SPD, Poisson equation |
| `vem2.mtx` | Virtual Element Method | Refined mesh 2 | Large sparse | SPD, high condition number |
| `spa1.mtx` | Structural Analysis | Structural mesh 1 | Sparse | Symmetric |
| `spa2.mtx` | Structural Analysis | Structural mesh 2 | Highly sparse | Symmetric |

The suite inspects structural properties:
- Spectral radius $\rho(B)$ of the iteration matrix
- Condition number $\kappa(A)$
- Non-zero density ($nnz / n^2$)
- Relative residual convergence: $\|b - A x_k\|_2 / \|b\|_2 \le \text{tol}$

---

## 🚀 Execution & Experiments

### 1. Installation
```bash
git clone https://github.com/mbroglio/sparse-linear-solvers.git
cd sparse-linear-solvers

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Benchmarks
- **Run Baseline Solvers**:
  ```bash
  python3 runner/run_baseline.py
  ```
- **Run Numba JIT Solvers**:
  ```bash
  python3 runner/run_numba.py
  ```
- **Analyze Matrix Properties**:
  ```bash
  python3 runner/matrix_analysis.py
  ```
- **Generate Convergence & Performance Plots**:
  ```bash
  python3 runner/generate_plots.py
  ```

---

## 📁 Repository Structure

```text
├── Assignment1_Report.pdf        # Full academic report (PDF)
├── README.md                     # Project documentation (English)
├── pyproject.toml                # Project configuration
├── requirements.txt              # Python dependencies (NumPy, SciPy, Numba, Matplotlib)
│
├── data/                         # Sparse Matrix Market test files (.mtx)
│   ├── vem1.mtx
│   ├── vem2.mtx
│   ├── spa1.mtx
│   └── spa2.mtx
│
├── linear_solvers/               # Core solver implementations
│   ├── baseline/                 # Vectorized SciPy/NumPy implementations
│   │   ├── base.py               # Abstract solver base class
│   │   └── iterative_methods.py  # Jacobi, Gauss-Seidel, Gradient, CG
│   └── numba/                    # JIT-compiled accelerated implementations
│       ├── base.py
│       └── iterative_methods.py  # Fast CSR loop solvers
│
└── runner/                       # Experiment runners and evaluation tools
    ├── run_baseline.py           # Baseline benchmark runner
    ├── run_numba.py              # Numba accelerated benchmark runner
    ├── matrix_analysis.py        # Matrix condition and spectral analysis
    ├── generate_plots.py         # Matplotlib convergence curve exporter
    ├── plot_results.py           # Comparative visualization tools
    └── results/                  # Stored metrics, CSVs, and generated plots
```

---

## 🎓 Authors & Academic Context

Project developed for the **Scientific Computing Methods** (*Metodi del Calcolo Scientifico*) course (Academic Year 2025/2026), Master's Degree in Computer Science, **Università degli Studi di Milano - Bicocca**:

- **Matteo Broglio** - Matricola `899562`
- **Lorenzo Caputo** - Matricola `894528`
- **Daniel Giuggioli** - Matricola `894415`

Full report available in [Assignment1_Report.pdf](Assignment1_Report.pdf).
