//! # Dendritic Machine Learning Crate
//!
//! Dendritic is a machine learning library for the Rust ecosystem.
//! This crate contains your standard machine learning algorithms and utilities for numerical computation.
//! There are multiple subcrates within this project that can be used to build machine learning models
//!
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.


//! # Published Crates

//! | Rust Crate                                                                  | Description                                                                            |
//! | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
//! | [dendritic_ndarray](https://crates.io/crates/dendritic-ndarray)             | N Dimensional array library for numerical computing                                    |
//! | [dendritic_datasets](https://crates.io/crates/dendritic-datasets)           | Variety of datasets for regression and classification tasks                                            |
//! | [dendritic_autodiff](https://crates.io/crates/dendritic-autodiff)           | Autodifferentiation crate for backward and forward operations                          |
//! | [dendritic_metrics](https://crates.io/crates/dendritic-metrics)             | Metrics package for measuring loss and activiation functions for non linear boundaries |
//! | [dendritic_preprocessing](https://crates.io/crates/dendritic-preprocessing) | Preprocessing library for normalization and encoding of data                           |
//! | [dendritic_bayes](https://crates.io/crates/dendritic-bayes)                 | Bayesian statistics package                                                            |
//! | [dendritic_clustering](https://crates.io/crates/dendritic-clustering)       | Clustering package utilizing various distance metrics                                  |
//! | [dendritic_knn](https://crates.io/crates/dendritic-knn)                     | K Nearest Neighbors for regression and classification                                  |
//! | [dendritic_models](https://crates.io/crates/dendritic-models)                  | Pre-trained models for testing `dendritic` functionality                               |
//! | [dendritic_regression](https://crates.io/crates/dendritic-regression)       | Regression package for linear modeling & multi class classification                    |
//! | [dendritic_trees](https://crates.io/crates/dendritic-trees)                 | Tree based models using decision trees and random forests                              |


//! ## Building The Dendritic Packages
//! Dendritic is made up of multiple indepedent packages that can be built separatley.
//! To install a package, add the following to your `Cargo.toml` file.

//! ```toml
//! [dependencies]
//! dendritic = { version = "<LATEST_VERSION>", features = ["bundled"] }
//! ```

//! ## Example IRIS Flowers Prediction
//! Down below is an example of using a multi class logstic regression model on the well known iris flowers dataset.
//! For more examples, refer to the `dendritic-models/src/main.rs` file. 

//! ```rust
//! use dendritic_datasets::iris::*;
//! use dendritic_regression::logistic::*;
//! use dendritic_metrics::loss::*;
//! use dendritic_metrics::activations::*;
//! use dendritic_preprocessing::encoding::*;


//! fn main() {

//!     // load data
//!     let data_path = "../../datasets/data/iris.parquet";
//!     let (x_train, y_train) = load_iris(data_path).unwrap();

//!     // encode the target variables
//!     let mut encoder = OneHotEncoding::new(y_train.clone()).unwrap();
//!     let y_train_encoded = encoder.transform();

//!     // create logistic regression model
//!     let mut log_model = MultiClassLogistic::new(
//!         &x_train,
//!         &y_train_encoded,
//!         softmax,
//!         0.1
//!     ).unwrap();

//!     log_model.sgd(500, true, 5);

//!     let sample_index = 100;
//!     let x_test = x_train.batch(5).unwrap();
//!     let y_test = y_train.batch(5).unwrap();
//!     let y_pred = log_model.predict(x_test[sample_index].clone());

//!     println!("Actual: {:?}", y_test[sample_index]);
//!     println!("Prediction: {:?}", y_pred.values());

//!     let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
//!     println!("LOSS: {:?}", loss);  
//! }
//! ```