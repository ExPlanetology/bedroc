<p align="center">
<img src="docs/logo.png" alt="bedroc logo" width="300"/>
</p>

# Bedroc

[![Release 0.1.0](https://img.shields.io/badge/Release-0.1.0-blue.svg)](https://github.com/ExPlanetology/atmodeller/releases/tag/v0.1.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/ExPlanetology/bedroc/actions/workflows/ci.yml/badge.svg)](https://github.com/ExPlanetology/bedroc/actions/workflows/ci.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Test coverage](https://img.shields.io/badge/Coverage-49%25-red)](https://github.com/ExPlanetology/bedroc)

## About
Bedroc&mdash;short for "Bayesian Engine for Determining Rock Origin and Composition"&mdash;is a Python package for applying Bayesian hierarchical modeling and probabilistic inference to research questions in geochemistry and petrology. It is designed to enable robust statistical analysis of complex datasets, accounting for uncertainty, partial pooling, and feature-wise variability. It is released under [The GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.en.html).

Although Bedroc originated within geochemistry and petrology, the underlying infrastructure is intentionally general. The package implements standard Bayesian modeling workflows&mdash;such as group-difference analysis, predictor-based models, uncertainty quantification, and hierarchical partial pooling&mdash;that are widely applicable across many scientific domains. As a result, Bedroc can be readily adapted for broader applications in Earth and planetary sciences, as well as other fields that benefit from principled statistical modeling and reproducible Bayesian inference.

Authors:

- Dan J. Bower

## Documentation

The documentation is available online, with options to download it in EPUB or PDF format:

[https://bedroc.readthedocs.io/en/latest/](https://bedroc.readthedocs.io/en/latest/)

## Quick install

Bedroc is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux). It is recommended to install Bedroc in a dedicated Python environment. Before installation, create and activate the environment, then run:

```pip install bedroc```

Downloading the source code is also recommended if you'd like access to the example notebooks in `notebooks/`.

## Citation

Please cite this GitHub repository and the specific Bedroc version number you used when using Bedroc in your research.

## Basic usage

Jupyter notebooks in the `notebooks/` directory demonstrate the core workflow of Bedroc. These notebooks serve as example scripts, typically using synthetic data to illustrate the basic principles of Bayesian analysis, including model construction, sampling, diagnostics, and plot generation.

Each notebook delegates most of the real work to the underlying functions and classes provided by the bedroc package. This keeps the examples clear and highlights how to connect the high-level workflow to the package internals.

The recommended workflow is to copy one of the notebooks and adapt it to your own needs. Since most of the underlying infrastructure is designed to accept NumPy arrays, the main modification you’ll need is to preprocess your own "real" data into an appropriate array format before passing it to the relevant functions.

Please note that bespoke or project-specific analysis scripts should not be committed back into this repository. This repository is intended to contain only the core, reusable, and generic infrastructure of Bedroc, keeping it clean, maintainable, and broadly applicable.