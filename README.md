# QEDFT on Hubbard Model

## Overview

This repository contains a "leveling project" designed to bridge the gap
between theoretical condensed matter physics and practical quantum
software engineering.

The primary objective is to implement **Quantum-Enhanced Density
Functional Theory (QEDFT)** for the **Fermi-Hubbard Model** (Hubbard Dimer
case, $L=2$). The project follows a strict "bottom-up" approach:

1. **Phase 0-1:** Manual construction of quantum operators and linear
   algebra backends (NumPy).
2. **Phase 2:** Exact diagonalization (FCI) to establish ground truth physics.
3. **Phase 3:** Classical Lattice DFT implementation (Self-Consistent Field loop).
4. **Phase 4:** Variational Quantum Eigensolver (VQE) implementation using Qiskit/PennyLane.

## Architecture

This project adheres to strict software engineering standards to ensure
reproducibility and modularity.

- **Basic Architecture:** Separation of input processing, core
  domain logic, and visualization.
- **Dependency Injection:** Mathematical backends (e.g., NumPy vs. Sparse)
  are injected into physics classes, keeping domain logic pure.
- **Type Safety:** Strict Python typing and generics.

## Getting Started

### Prerequisites

- Python 3.13+
- `uv` (Project Manager)

### Installation

Clone the repository and sync dependencies:

```bash
git clone https://github.com/KaueMiziara/dft-vqe-hubbard.git
cd dft-vqe-hubbard
uv sync
```

### Running Tests

We use `pytest` for unit testing and regression checks.

The project is configured with `pre-commit` hooks
to prevent broken commits.

```bash
uv run pytest
```

## Methodology

### Phase 1: Algebra & Operators (Completed)

- **Goal:** Construct the computational basis and operators
  without high-level quantum frameworks.
- **Implementation:** Generic `OperatorBackend` with a concrete `NumpyBackend`.
- **Physics:** Manual implementation of the Jordan-Wigner
  mapping ($c^\dagger \to P_k \otimes \dots$) to build the Number Operator $\hat{n}$.

### Phase 2: Exact Solution (Current)

- **Goal:** Solve the Hubbard Hamiltonian exactly ($H = H_{kin} + H_{int}$) to
  observe the Mott transition.
- **Stack:** `numpy.linalg.eigh` for full diagonalization.
