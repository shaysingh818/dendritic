//! # Dendritic 
//!
//! **Dendritic** is a lightweight and extensible optimization library built with flexibility in mind.
//! Contains utilities for first order optimization with multi variate/vector valued functions using `ndarray`.
//! This crate aims to contain extensible interfaces for common abstractions in optimization & machine learning problems.


//! ## 🚀 Features
//! 
//! - 📐 **Auto-Differentiation**: Reverse-mode autodiff for computing gradients using `ndarray`.
//! - ⚙️ **Optimizers**: Built-in optimizers like SGD, Adam, etc.
//! - 📈 **Regression Models**: Traditional regression models (Linear, Logistic).
//! - 🔣 **Preprocessing**: Lightweight utilities for common preprocessing tasks (e.g., one-hot encoding).
//! - 🧱 **Modular**: Designed to be flexible and easy to extend for research or custom pipelines.

//! ## 🔮 Future Enhancements
//! 
//! There are more features on the roadmap for this crate. Version `v2` was a redesign focused on improving crate structure and aligning functionality with abstractions common in optimization theory.
//!
//! Below are some ideas for future features that may be incorporated into the crate:
//!
//! - **Second Order Optimization**  
//!   Methods like Newton's or Secant methods that leverage second-order derivatives (Hessian) for faster convergence.
//!
//! - **Population Methods**  
//!   Optimization techniques that maintain and evolve a population of candidate solutions. Includes genetic algorithms and evolutionary strategies.
//!
//! - **Zero Order Methods**  
//!   Approaches that do not require gradient information, useful in settings where derivatives are unavailable or expensive to compute.

pub mod autodiff;
pub mod optimizer;
pub mod preprocessing;
