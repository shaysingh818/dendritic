# Dendritic
[![Downloads](https://img.shields.io/crates/d/dendritic)](https://img.shields.io/crates/d/dendritic)
[![Build Status](https://github.com/shaysingh818/Dendritic/actions/workflows/pipeline.yml/badge.svg)](https://github.com/shaysingh818/Dendritic/actions)
[![dependency status](https://deps.rs/repo/github/shaysingh818/Dendritic/status.svg)](https://deps.rs/repo/github/shaysingh818/dendritic)
[![codecov](https://codecov.io/gh/shaysingh818/Dendritic/branch/main/graph/badge.svg?token=0xV88q8KU0)](https://codecov.io/gh/shaysingh818/denritic)
[![Latest Version](https://img.shields.io/crates/v/dendritic.svg)](https://crates.io/crates/dendritic)
[![Docs](https://img.shields.io/badge/docs.rs-denritic-green)](https://docs.rs/dendritic)

**Dendritic** is a lightweight and extensible optimization library built with flexibility in mind. Contains utilities for first order optimization with multi variate/vector valued functions using `ndarray`. This crate aims to contain extensible interfaces for common abstractions in optimization & machine learning problems. 
## 🚀 Features

- 📐 **Auto-Differentiation**: Reverse-mode autodiff for computing gradients using ndarray.
- ⚙️ **Optimizers**: Built-in optimizers like SGD, Adam etc. 
- 📈 **Regression Models**: Traditional regression models (Linear, Logistic)
- 🔣 **Preprocessing**: Lightweight utilities for common preprocessing tasks (e.g., one-hot encoding).
- 🧱 **Modular**: Designed to be flexible and easy to extend for research or custom pipelines.

## Future Enhancements

There are more features on the roadmap for this crate, `v2` of this crate was a redesign of the crate structure and honing in on features that are more aligned with common abstractions in optimization theory. Down below are some ideas for future features that will be incorporated into this crate.

| Feature                       | Description                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Second Order Optimization** | Using newton/secant methods that leverage the second derivative for faster convergence.                             |
| **Population Methods**        | Algorithms that involve a "population" of design points to iterate on and converge towards. Genetic algorithms etc. |
| **Zero Order Methods**        | Optimization methods that don't rely on the first or second derivative for finding the local max or minimum.        |




