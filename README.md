# Dendritic
[![Downloads](https://img.shields.io/crates/d/duckdb)](https://img.shields.io/crates/d/duckdb)
[![Build Status](https://github.com/wangfenjin/duckdb-rs/workflows/CI/badge.svg)](https://github.com/wangfenjin/duckdb-rs/actions)
[![dependency status](https://deps.rs/repo/github/wangfenjin/duckdb-rs/status.svg)](https://deps.rs/repo/github/wangfenjin/duckdb-rs)
[![codecov](https://codecov.io/gh/wangfenjin/duckdb-rs/branch/main/graph/badge.svg?token=0xV88q8KU0)](https://codecov.io/gh/wangfenjin/duckdb-rs)
[![Latest Version](https://img.shields.io/crates/v/duckdb.svg)](https://crates.io/crates/duckdb)
[![Docs](https://img.shields.io/badge/docs.rs-duckdb-green)](https://docs.rs/duckdb)

Dendrite is a general purpose supervised/un-supervised machine learning library written for the rust ecosystem. It contains the required data structures & algorithms needed for general machine learning. It acts as core library with packages for predictive data modeling.

# Disclaimer
The dendritic project is a toy machine learning library built for learning and research purposes.
It is not advised by the maintainer to use this library as a production ready machine learning library.
This is a project that is still very much a work in progress.

# Published Packages

| Rust Crate                | Description                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `dendritic_autodiff`      | Autodifferentiation crate for backward and forward operations                          |
| `dendritic_bayes`         | Bayesian statistics package                                                            |
| `dendritic_clustering`    | Clustering package utilizing various distance metrics                                  |
| `dendritic_datasets`      | Combination of lasso and ridge regression                                              |
| `dendritic_knn`           | K Nearest Neighbors for regression and classification                                  |
| `dendritic_metrics`       | Metrics package for measuring loss and activiation functions for non linear boundaries |
| `dendritic_models`        | Pre-trained models for testing `dendritic` functionality                               |
| `dendritic_ndarray`       | N Dimensional array library for numerical computing                                    |
| `dendritic_preprocessing` | Preprocessing library for normalization and encoding of data                           |
| `dendritic_regression`    | Regression package for linear modeling & multi class classification                    |
| `dendritic_trees`         | Tree based models using decision trees and random forests                              |

## Building The Dendritic Packages
Dendritic is made up of multiple indepedent packages that can be built separatley.
To install a package, add the following to your `Cargo.toml` file.

```toml
[dependencies]
# Assume that version Dendritic version 1.1.0 is used.
dendritic_regression = { version = "1.1.0", features = ["bundled"] }
```

## Example IRIS Flowers Prediction

```rust
use datasets::iris::*;
use regression::logistic::*;
use metrics::loss::*;
use metrics::activations::*;
use preprocessing::encoding::*;

// load data
let (x_train, y_train) = load_iris().unwrap();

// encode the target variables
let mut encoder = OneHotEncoding::new(y_train.clone()).unwrap();
let y_train_encoded = encoder.transform();

// create logistic regression model
let mut log_model = MultiClassLogistic::new(
    x_train.clone(),
    y_train_encoded.clone(),
    softmax,
    0.1
).unwrap();

log_model.sgd(500, true, 5);

let sample_index = 50;
let x_test = x_train.batch(5).unwrap();
let y_test = y_train.batch(5).unwrap();
let y_pred = log_model.predict(x_test[sample_index].clone());

println!("Actual: {:?}", y_test[sample_index]);
println!("Prediction: {:?}", y_pred.values());

let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
println!("LOSS: {:?}", loss);  
```




