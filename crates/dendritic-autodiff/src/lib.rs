
//! # Dendritic Autodifferentiation Crate
//!
//! This crate allows for autodifferentiation to be performed during backpropogation of some ML algorithms.
//! The autodiff library currently supports operations for weight regularization, dot product and elementwise operations.
//! This crate serves as the base depedencies for most of the algorithms in the regression package
//!
//! ## Features
//! - **Node**: Node structure for holding shared methods across all values in a computation graph.
//! - **Ops**: Operations with forward and backward pass implemented
//! - **Regularizers**: Operations specific to weight regualarization to prevent overfitting
//!
//! ## Example Usage
//! This is an example of creating the computation graph for a linear operation

//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_autodiff::node::*; 
//! use dendritic_autodiff::ops::*;
//! use dendritic_autodiff::regularizers::*; 
//!
//! fn main() {

//!     //Load saved ndarrays (model parameters)
//!     let x_path = "data/linear_modeling_data/inputs"; 
//!     let w_path = "data/linear_modeling_data/weights";
//!     let b_path = "data/linear_modeling_data/bias";
//!
//!     let x: NDArray<f64> = NDArray::load(x_path).unwrap();
//!     let w: NDArray<f64> = NDArray::load(w_path).unwrap();
//!     let b: NDArray<f64> = NDArray::load(b_path).unwrap();
//!
//!     // Convert ndarrays to value nodes
//!     let inputs = Value::new(&x);
//!     let weights = Value::new(&w);
//!     let bias = Value::new(&b);
//!
//!     // Create computation graph for linear layer
//!     let mut linear= ScaleAdd::new(
//!         Dot::new(inputs.clone(), weights.clone()),
//!         bias
//!     );
//!
//!     linear.forward(); // perform forward pass
//! }
//! ```
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod node;
pub mod ops;
pub mod regularizers; 
