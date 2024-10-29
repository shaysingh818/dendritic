# Dendritic
[![Downloads](https://img.shields.io/crates/d/dendritic)](https://img.shields.io/crates/d/dendritic)
[![Build Status](https://github.com/shaysingh818/Dendrite/actions/workflows/pipeline.yml/badge.svg)](https://github.com/shaysingh818/Dendrite/actions)
[![dependency status](https://deps.rs/repo/github/shaysingh818/Dendrite/status.svg)](https://deps.rs/repo/github/shaysingh818/Dendrite)
[![codecov](https://codecov.io/gh/wangfenjin/duckdb-rs/branch/main/graph/badge.svg?token=0xV88q8KU0)](https://codecov.io/gh/wangfenjin/duckdb-rs)
[![Latest Version](https://img.shields.io/crates/v/dendritic.svg)](https://crates.io/crates/dendritic)
[![Docs](https://img.shields.io/badge/docs.rs-denritic-green)](https://docs.rs/dendritic)

Dendrite is a general purpose supervised/un-supervised machine learning library written for the rust ecosystem. It contains the required data structures & algorithms needed for general machine learning. It acts as core library with packages for predictive data modeling.

# Disclaimer
The dendritic project is a toy machine learning library built for learning and research purposes.
It is not advised by the maintainer to use this library as a production ready machine learning library.
This is a project that is still very much a work in progress.

# Published Crates

| Rust Crate                                                                  | Description                                                                            |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [dendritic_autodiff](https://crates.io/crates/dendritic-autodiff)           | Autodifferentiation crate for backward and forward operations                          |
| [dendritic_bayes](https://crates.io/crates/dendritic-bayes)                 | Bayesian statistics package                                                            |
| [dendritic_clustering](https://crates.io/crates/dendritic-clustering)       | Clustering package utilizing various distance metrics                                  |
| [dendritic_datasets](https://crates.io/crates/dendritic-datasets)           | Combination of lasso and ridge regression                                              |
| [dendritic_knn](https://crates.io/crates/dendritic-knn)                     | K Nearest Neighbors for regression and classification                                  |
| [dendritic_metrics](https://crates.io/crates/dendritic-metrics)             | Metrics package for measuring loss and activiation functions for non linear boundaries |
| [dendritic_models]()                                                        | Pre-trained models for testing `dendritic` functionality                               |
| [dendritic_ndarray](https://crates.io/crates/dendritic-ndarray)             | N Dimensional array library for numerical computing                                    |
| [dendritic_preprocessing](https://crates.io/crates/dendritic-preprocessing) | Preprocessing library for normalization and encoding of data                           |
| [dendritic_regression](https://crates.io/crates/dendritic-regression)       | Regression package for linear modeling & multi class classification                    |
| [dendritic_trees](https://crates.io/crates/dendritic-trees)                 | Tree based models using decision trees and random forests                              |

## Building The Dendritic Packages
Dendritic is made up of multiple indepedent packages that can be built separatley.
To install a package, add the following to your `Cargo.toml` file.

```toml
[dependencies]
# Assume that version Dendritic version 1.1.0 is used.
dendritic_regression = { version = "1.1.0", features = ["bundled"] }
```

## Example IRIS Flowers Prediction
Down below is an example of using a multi class logstic regression model on the well known iris flowers dataset.
For more examples, refer to the `dendritic-models/src/main.rs` file. 

```rust
use datasets::iris::*;
use regression::logistic::*;
use metrics::loss::*;
use metrics::activations::*;
use preprocessing::encoding::*;


fn main() {

    // load data
    let data_path = "../../datasets/data/iris.parquet";
    let (x_train, y_train) = load_iris(data_path).unwrap();

    // encode the target variables
    let mut encoder = OneHotEncoding::new(y_train.clone()).unwrap();
    let y_train_encoded = encoder.transform();

    // create logistic regression model
    let mut log_model = MultiClassLogistic::new(
        &x_train,
        &y_train_encoded,
        softmax,
        0.1
    ).unwrap();

    log_model.sgd(500, true, 5);

    let sample_index = 100;
    let x_test = x_train.batch(5).unwrap();
    let y_test = y_train.batch(5).unwrap();
    let y_pred = log_model.predict(x_test[sample_index].clone());

    println!("Actual: {:?}", y_test[sample_index]);
    println!("Prediction: {:?}", y_pred.values());

    let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
    println!("LOSS: {:?}", loss);  
}
```




