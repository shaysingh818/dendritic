//! # Dendritic Regression Crate
//!
//! This crate contains functionality for performing regression with linear logistic models.
//! Contains standard linear regression methods with weight regularization and logistic regression.
//! The categorization of these models is subject to change as this project moves forward.
//! This may eventually just become a "linear" modeling package.
//!
//! ## Features
//! - **Linear**: Standard scalar and min max normlization of data.
//! - **Lasso**: One hot encoding for multi class data
//! - **Ridge**: One hot encoding for multi class data
//! - **Elastic Net**: One hot encoding for multi class data
//! - **Logistic**: One hot encoding for multi class data
//!
//! ## Example Linear Model Usage
//! This is an example of using the linear models available in the regression crate for dendritic. 
//! The examples will contain use of `Linear`, `Ridge`, `Lasso` and `ElasticNet`
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_metrics::loss::*;
//! use dendritic_regression::elastic_net::*;
//! use dendritic_regression::linear::*;
//! use dendritic_regression::ridge::*;
//! use dendritic_regression::lasso::*;
//! use dendritic_datasets::airfoil_noise::*;
//! 
//! fn main() {
//!
//!    // Hyperparameters
//!    let learning_rate: f64 = 0.01;
//!    let lambda: f64 = 0.001;

//!    let data_path = "../dendritic-datasets/data/airfoil_noise_data.parquet";
//!    let (x_train, y_train) = load_airfoil_data(data_path).unwrap();
//!
//!    // linear
//!    let mut linear = Linear::new(
//!        &x_train, 
//!        &y_train, 
//!        0.01
//!    ).unwrap();
//!
//!    // ridge
//!    let mut ridge = Ridge::new(
//!        &x_train, 
//!        &y_train,
//!        lambda, learning_rate
//!    ).unwrap();
//!
//!    // lasso
//!    let mut lasso = Lasso::new(
//!        &x_train, 
//!        &y_train,
//!        lambda, learning_rate
//!    ).unwrap();
//!
//!    // elastic net
//!    let mut model = ElasticNet::new(
//!        &x_train, 
//!        &y_train,
//!        lambda, learning_rate
//!    ).unwrap();
//!
//!    // Example of training the linear model
//!    model.train(1000, false); // train for 1000 epochs (logging set to false)
//!    let outputs = model.predict(x_train);
//!    let loss = mse(&outputs, &y_train).unwrap(); 
//!    println!("Output: {:?}", outputs);
//!    println!("Loss: {:?}", loss)
//! }
//! ```

//! ## Example Logistic Model Usage
//! This is an example of using the logistic regression model provided by dendritic.
//! The example below uses binary classification, but multi class is also supported with `MultiClassLogistic`.
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_metrics::loss::*;
//! use dendritic_regression::logistic::*;
//! use dendritic_datasets::breast_cancer::*;
//! use dendritic_metrics::activations::*;
//! 
//! fn main() {
//!
//!    // load data 
//!    let data_path = "../dendritic-datasets/data/breast_cancer.parquet";
//!    let (x_train, y_train) = load_breast_cancer(data_path).unwrap();
//!
//!    // create logistic regression model
//!    let mut log_model = Logistic::new(
//!        &x_train,
//!        &y_train,
//!        sigmoid_vec,
//!        0.001
//!    ).unwrap();
//!
//!    log_model.sgd(1000, true, 5);
//!    let sample_index = 450;
//!    let x_test = x_train.batch(5).unwrap();
//!    let y_test = y_train.batch(5).unwrap();
//!    let y_pred = log_model.predict(x_test[sample_index].clone());
//!    println!("Actual: {:?}", y_test[sample_index]);
//!    println!("Prediction: {:?}", y_pred.values());
//!
//!    let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
//!    println!("LOSS: {:?}", loss);
//!
//! }
//! ```
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod logistic;
pub mod linear;
pub mod ridge;
pub mod lasso;
pub mod elastic_net;
