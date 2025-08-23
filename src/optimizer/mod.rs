//! # Optimizer & model abstractions with defaults
//!
//! This is a crate focused on abstractions for common optimization algorithms.
//! Currently supports optimizers build on SGD (stochastic gradient descent). 
//! Standard linear & logistic regression models are implemented using the optimizer abstractions.
//! Houses abstractions for models, training, and evaluation,
//! All parameters currently only support `ndarrays` and the `f64` type. 
//!
//! ## Features
//! - **Model/Optimizer Abstractions**: Contains interfaces for creating models and optimizers. 
//! - **Train**: Utilities for training loops for single datasets or batches of data with optimizers.
//! - **Regression**: Suite of linear & logistic regression models.
//! - **Registry**: Operation registry for managing and looking up operations. 
//! - **Default Operations**: Suite of default arithmetic, activation, and loss functions.
//!
//! # Example: Linear Regression with SGD + Adam Optimizer
//!
//! This example demonstrates how to train a simple linear regression model
//! using stochastic gradient descent (SGD) combined with the Adam optimizer.
//!
//! The example uses synthetic data to fit a linear model:
//!
//! ```rust
//! use ndarray::{arr2, Array2};
//!
//! use dendritic::optimizer::model::*;
//! use dendritic::optimizer::optimizers::*;
//! use dendritic::optimizer::optimizers::Optimizer;
//! use dendritic::optimizer::regression::sgd::*;
//!
//! // Sample training data
//! fn load_sample_data() -> (Array2<f64>, Array2<f64>) {
//!     let x = arr2(&[
//!         [1.0, 2.0, 3.0],
//!         [2.0, 3.0, 4.0],
//!         [3.0, 4.0, 5.0],
//!         [4.0, 5.0, 6.0],
//!         [5.0, 6.0, 7.0]
//!     ]);
//!
//!     let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);
//!     (x, y)
//! }
//!
//! fn main() {
//!     let alpha = 0.1; // Learning rate
//!
//!     // Load data and initialize model
//!     let (x, y) = load_sample_data();
//!     let mut model = SGD::new(&x, &y, alpha).unwrap();
//!     let mut optimizer = Adam::default(&model);
//!
//!     // Train the model
//!     for _ in 0..350 {
//!         model.graph.forward();
//!         model.graph.backward();
//!         optimizer.step(&mut model);
//!     }
//!
//!     // Retrieve and print loss and predictions
//!     let loss_total = model.loss();
//!     let predicted = model.predicted().mapv(|x| x.round());
//!
//!     println!("Loss: {:?}", loss_total);
//!     println!("Predictions: {:?}", model.predicted());
//! }
//! ```

pub mod model;
pub mod train;
pub mod regression;
pub mod optimizers; 
