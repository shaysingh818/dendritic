//! # Dendritic Metrics Crate
//!
//! This crate contains metrics for measuring loss, accuracy of general ML models available for dendritic.
//! Metrics contain loss and activiation functions.
//!
//! ## Features
//! - **Activations**: Activation functions for non linear data.
//! - **Loss**: Loss functions for measuring accuracy of classifiers/regressors
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-metrics = "0.1"
//! ```
//! ## Example Usage
//! This is an example of some of the loss and activation functions dendritic has to offer
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_metrics::activations::*; 
//! use dendritic_metrics::loss::*; 
//! 
//! fn main() {
//!
//!     // Mocked Prediction values
//!     let y_pred: NDArray<f64> = NDArray::array(
//!         vec![10, 1],
//!         vec![
//!             0.0, 0.0, 1.0, 0.0, 1.0,
//!             1.0, 1.0, 1.0, 1.0, 1.0
//!         ]
//!      ).unwrap();
//!
//!      // Mocked true values
//!      let y_true: NDArray<f64> = NDArray::array(
//!         vec![10, 1],
//!         vec![
//!             0.19, 0.33, 0.47, 0.7, 0.74,
//!             0.81, 0.86, 0.94, 0.97, 0.99
//!         ]
//!      ).unwrap();
//!
//!      // Calculate binary cross entropy for predicted and true values
//!      let result = binary_cross_entropy(&y_true, &y_pred).unwrap();
//!      println!("{:?}", result); 
//!
//!      // Input dataset to perform softmax activation
//!      let input: NDArray<f64> = NDArray::array(
//!         vec![3, 1],
//!         vec![1.0, 1.0, 1.0]
//!      ).unwrap();
//!
//!      let sm_result = softmax_prime(input);
//!      println!("{:?}", sm_result.values()); 
//! }
//! ```
pub mod loss;
pub mod activations;
pub mod utils;
