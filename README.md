# ProbGraph: Exploring $\epsilon$ in Stochastic Interpolants

**How much noise is enough?** This project explores the role of the diffusion coefficient $\epsilon$ in stochastic interpolants, investigating the trade-off between fidelity (preserving content) and robustness/diversity (exploring without collapsing) in score-based generative models.

To get started, just run `./launch.sh`.

**Implementation details:** This project is built on top of the [`interflow`](https://github.com/malbergo/stochastic-interpolants) module from the [stochastic-interpolants repository](https://github.com/malbergo/stochastic-interpolants) by Albergo et al. We cloned the core `interflow` implementation and built our experimental framework around it, including custom interpolants, noise schedules, and extensive experiments on CelebA. This work represents approximately **40 hours of development and experimentation** to achieve the reported results.


## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [Tests](#tests)
- [License](#license)
- [Contact](#contact)

## About

This project implements **stochastic interpolants**, a unifying framework connecting deterministic dynamics (flow matching) and stochastic dynamics (diffusion). Stochastic interpolants generalize score-based generative models by separating deterministic transport from stochasticity, allowing precise control over the trade-off between fidelity, diversity, and numerical stability via the diffusion coefficient $\epsilon$.

The main experimental question is: **how much noise $\epsilon$ do we really need?** We explore different interpolant paths $I(t,\cdot,\cdot)$, noise schedules $\gamma(t)$, and diffusion levels $\epsilon$ to understand how $\epsilon$ controls the fidelity vs robustness/diversity trade-off.

**Main file to run:** [`notebooks/cifar/cifar10lsun_patched_interpolants_epsilon.py`](https://github.com/janisaiad/probgraph/blob/master/notebooks/cifar/cifar10lsun_patched_interpolants.py)


**Poster:** See [`refs/poster/conference_poster_4.pdf`](refs/poster/conference_poster_4.pdf) for a detailed overview of the project.

## Installation

To install dependencies using uv, follow these steps:

1. Install uv:
   
   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, you can install uv using:
   - pipx (recommended): `pipx install uv`
   - pip: `pip install uv`
   - Homebrew: `brew install uv`
   - WinGet: `winget install --id=astral-sh.uv -e`
   - Scoop: `scoop install main/uv`

2. Using uv in this project:

   - Initialize a new virtual environment:
   ```bash
   uv venv
   ```

   - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

   - Install dependencies from requirements.txt:
   ```bash
   uv add -r requirements.txt
   ```


   - Add a new package:
   ```bash
   uv add package_name
   ```

   - Remove a package:
   ```bash
   uv remove package_name
   ```

   - Update a package:
   ```bash
   uv pip install --upgrade package_name
   ```

   - Generate requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

   - List installed packages:
     ```bash
     uv pip list
     ```

## Usage

The main file to run is:
```bash
python notebooks/cifar/cifar10lsun_patched_interpolants.py
```

Or use the [Colab notebook](https://colab.research.google.com/github/janisaiad/probgraph/blob/master/notebooks/cifar/cifar10lsun_patched_interpolants_epsilon.ipynb) for an interactive experience.

This script implements stochastic interpolants with different $\epsilon$ values to explore the trade-off between fidelity and diversity in generative modeling.

## Warning

If you're using macOS or Python 3, replace `pip` with `pip3` in line 1 of ```launch.sh```

Replace with your project folder name (which means the name of the library you are deving) in :```tests/test_env.py: ```