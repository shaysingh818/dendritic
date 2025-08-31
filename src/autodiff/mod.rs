//! # Autodifferentiation Abstractions
//!
//! This is a crate focused on automatic differentation & working with derivatives in multiple dimensions.
//! The autodiff library currently supports operations for simple arithmetic and applying common activation functions. 
//! This crate aims to have a simple framework for creating differentiable functions that minimize
//! or maximize another function that measures loss. Operations for autodifferentation are currently 
//! only supporting ndarrays and f64 values. 
//!
//! ## Features
//! - **Tensor Values**: This crate supports creating values in multiple dimensions. This achieved
//! using the `ndarray` crate for rust. 
//! - **Nodes**: Contains structure for storing nodes (operations). Nodes contain shared routines
//! that can be extended for different types of operations.
//! - **Graph**: General graph utility that stores the relationships of operations.
//! - **Registry**: Operation registry for managing and looking up operations. 
//! - **Default Operations**: Suite of default arithmetic, activation, and loss functions.
//!
//! ## 🧪 Example: Linear Regression with Gradient Descent
//!
//! This example demonstrates how to use the `dendritic` crate to build and optimize a simple
//! linear regression model using auto-differentiation and manual parameter updates.
//!
//! ```rust
//! use chrono::Local;
//! use ndarray::{arr2, Array2};
//!
//! use dendritic::autodiff::graph::*;
//! use dendritic::autodiff::operations::activation::*;
//! use dendritic::autodiff::operations::arithmetic::*;
//! use dendritic::autodiff::operations::loss::*;
//!
//! fn main() {
//!     let lr: f64 = 0.01;
//!
//!     let w = Array2::<f64>::zeros((3, 1));
//!     let b = Array2::<f64>::zeros((1, 1));
//!
//!     let x = arr2(&[
//!         [1.0, 2.0, 3.0],
//!         [2.0, 3.0, 4.0],
//!         [3.0, 4.0, 5.0],
//!         [4.0, 5.0, 6.0],
//!         [5.0, 6.0, 7.0]
//!     ]);
//!
//!     let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);
//!
//!     let mut graph = ComputationGraph::new();
//!     graph.mul(vec![x, w]);       // node 0
//!     graph.add(vec![b]);          // node 1
//!     graph.mse(y.clone());        // node 2
//!
//!     // Mark parameter nodes (manual for now)
//!     graph.add_parameter(1);      // w
//!     graph.add_parameter(3);      // b
//!
//!     for _epoch in 0..1000 {
//!         graph.forward();
//!
//!         let loss_node = graph.node(6);
//!         let loss = loss_node.output();
//!         println!("Loss: {:?}", loss.as_slice().unwrap());
//!
//!         graph.backward();
//!
//!         for var_idx in graph.parameters() {
//!             let var = graph.node(var_idx);
//!             let grad = var.grad() * (lr / y.len() as f64);
//!             let delta = var.output() - grad;
//!             graph.mut_node_output(var_idx, delta.clone());
//!         }
//!     }
//! }
//! ```
//! # 🧮 Supported Operations
//!
//! Dendritic provides a rich set of built-in operations that work out-of-the-box.
//!
//! - No need to implement traits — these are registered internally and ready to use.
//! - All default operations are stored in the **operation registry**, making them available for serialization and model export.
//!
//! ## ➗ Arithmetic
//!
//! - **Add**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Adds one or more values. Supports both binary and unary addition.
//!
//! - **Sub**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Subtracts one or more values. Supports both binary and unary subtraction.
//!
//! - **Mul**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Multiplies values. Supports scalar multiplication and dot products.
//!
//! ## 🧠 Activation
//!
//! - **Sigmoid**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Applies the sigmoid activation function element-wise.
//!
//! - **Tanh**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Applies the hyperbolic tangent function element-wise.
//!
//! ## 📉 Loss
//!
//! - **MSE (Mean Squared Error)**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Standard loss for regression tasks.
//!
//! - **BinaryCrossEntropy**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Binary classification loss using cross-entropy.
//!
//! - **CategoricalCrossEntropy**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: Multi-class classification loss using cross-entropy.
//!
//! - **DefaultLossFunction**
//!   - Types: `Array2<f64>`, `f64`
//!   - Description: A placeholder loss used for prototyping models when the final loss function is unknown.
pub mod node;
pub mod operations;
pub mod tensor;
pub mod graph;
pub mod registry;
pub mod prelude; 


