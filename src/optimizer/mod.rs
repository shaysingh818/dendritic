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
//!
//! # Example: Binary & Multi-Class Logistic Regression
//!
//! This example demonstrates how to use the `dendritic` crate to train both binary and
//! multi-class logistic regression models using `ndarray` data structures.
//!
//! The example covers:
//!
//! - Creating training data
//! - Training logistic regression models
//! - Saving and loading models
//! - Making predictions
//!
//! ```rust
//! use std::fs;
//! use std::fs::File;
//!
//! use ndarray::{arr2, Array2};
//! use dendritic::optimizer::model::*;
//! use dendritic::optimizer::train::*;
//! use dendritic::optimizer::regression::logistic::*;
//!
//! fn load_multi_class() -> (Array2<f64>, Array2<f64>) {
//!     let x = arr2(&[
//!         [1.0, 2.0],
//!         [1.5, 1.8],
//!         [2.0, 1.0], // Class 0
//!         [4.0, 4.5],
//!         [4.5, 4.8],
//!         [5.0, 5.2], // Class 1
//!         [7.0, 7.5],
//!         [7.5, 8.0],
//!         [8.0, 8.5], // Class 2
//!     ]);
//!
//!     let y = arr2(&[
//!         [1.0, 0.0, 0.0],
//!         [1.0, 0.0, 0.0],
//!         [1.0, 0.0, 0.0],
//!         [0.0, 1.0, 0.0],
//!         [0.0, 1.0, 0.0],
//!         [0.0, 1.0, 0.0],
//!         [0.0, 0.0, 1.0],
//!         [0.0, 0.0, 1.0],
//!         [0.0, 0.0, 1.0],
//!     ]);
//!
//!     (x, y)
//! }
//!
//! fn load_binary_data() -> (Array2<f64>, Array2<f64>) {
//!     let x = arr2(&[
//!         [1.0, 2.0],
//!         [2.0, 1.0],
//!         [1.5, 1.8],
//!         [3.0, 3.2],
//!         [2.8, 3.0],
//!         [5.0, 5.5],
//!         [6.0, 5.8],
//!         [5.5, 6.0],
//!         [6.2, 5.9],
//!         [7.0, 6.5],
//!     ]);
//!
//!     let y = arr2(&[
//!         [0.0],
//!         [0.0],
//!         [0.0],
//!         [0.0],
//!         [0.0],
//!         [1.0],
//!         [1.0],
//!         [1.0],
//!         [1.0],
//!         [1.0],
//!     ]);
//!     (x, y)
//! }
//!
//! fn main() -> std::io::Result<()> {
//!     // Binary logistic regression
//!     let (x, y) = load_binary_data();
//!     let mut model = Logistic::new(&x, &y, false, 0.01).unwrap();
//!
//!     // Multi-class logistic regression
//!     let (x1, y1) = load_multi_class();
//!     let mut multi_class_model = Logistic::new(&x1, &y1, true, 0.01).unwrap();
//!
//!     // Train and save logistic model
//!     model.train(1000);
//!     model.save("data/logistic")?;
//!
//!     // Train and save multi-class logistic model
//!     multi_class_model.train(2000);
//!     multi_class_model.save("data/multiclass_logistic")?;
//!
//!     // Load the saved model and make predictions
//!     let mut loaded = Logistic::load("data/multiclass_logistic").unwrap();
//!     let output = loaded.predict(&x1);
//!
//!     println!("Class Predictions: {:?}", output);
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod train;
pub mod regression;
pub mod optimizers; 
