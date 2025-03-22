
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
//!
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod node;
pub mod ops;
pub mod tensor;
pub mod graph;
pub mod binary; 
pub mod unary; 
