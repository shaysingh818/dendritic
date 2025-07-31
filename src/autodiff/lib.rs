
//! # Dendritic Autodifferentiation Crate
//!
//! This is a crate focused on automatic differentation & working with derivatives in multiple dimensions.
//! The autodiff library currently supports operations for simple arithmetic and applying common activation functions. 
//! This crate aims to have a simple framework for creating differentiable functions that minimize
//! or maximize another function that measures loss.
//!
//! ## Features
//! - **Tensor Values**: This crate supports creating values in multiple dimensions. This achieved
//! using the `ndarray` crate for rust. 
//! - **Nodes**: Contains structure for storing nodes (operations). Nodes contain shared routines
//! that can be extended for different types of operations.
//! - **Graph**: General graph utility that stores the relationships of operations.
//!
//! ## Example Usage
//! This is an example of creating the computation graph for a linear operation
//!
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod node;
pub mod operations;
pub mod tensor;
pub mod graph;
pub mod error;
pub mod registry; 
