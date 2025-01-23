# chaogatenn

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)

### Code for gradient based optimization of chaogates paper
This code base is uv compatible and pip installable.

### Authors
Anil Radhakrishnan, Sudeshna Sinha, K. Murali ,William L. Ditto

### [Link to paper](https://www.sciencedirect.com/science/article/pii/S0960077925000207)

### Key Results
- A gradient-based optimization framework for tuning chaotic systems to match predefined logic gate behavior.
- Extension of the framework to show simultaneous optimization of multiple logic gates for logic circuits like the half-adder.
- A demonstration and comparison of the reconfigurability of chaogates across nonlinear map configurations, showing the efficacy of using the same nonlinear system to perform multiple gate operations through parameter tuning

### Installation
We recommend using [uv](https://docs.astral.sh/uv/) to manage python and install the package.

Then, you can simply git clone the repository and run,

```bash
uv pip install .
```
to install the package with all dependencies.

### Usage

The notebooks in the `nbs` illustrate different extensions and tests of the chaogates framework.

The scripts in the `scripts` directory are the same as the `Diff_chao_config` notebooks but with argparsing for easy command line usage for use in batch processing.
To run the scripts, you can use the `uv run` command to run the scripts in the `scripts` directory.
The bash scripts in the `scripts` directory can be used to run the scripts in batch mode.

The analysis of the statistical run results can be done using the `analysis` amd `plotter` notebooks in the `nbs` directory.


### Code References
- [Equinox](https://docs.kidger.site/equinox/) Pytorch like module for JAX
- [JAX](https://github.com/jax-ml/jax) Accelerator-oriented array computation and program transformation
- [Optax](https://github.com/google-deepmind/optax) Gradient processing and optimization library for JAX
