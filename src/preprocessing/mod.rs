//! # Data pre-processing module
//!
//! This module provides utilities for data preprocessing tasks commonly encountered in machine learning workflows.
//! It includes functions for data normalization, encoding categorical variables, and handling missing values.
//! The goal is to prepare raw data for training machine learning models by transforming it into a suitable format.
//! All parameters currently only support `ndarrays` and the `f64` type. 
//!
//! ## Features
//! - **Standard Scalar**: Functions for normalizing input features to a standard scale.
//! - **Min Max Scalar**: Utilities for scaling features to a specific range (e.g., [0, 1]).
//! - **One Hot Encoding**: Strategies for converting categorical variables into a binary matrix.
//!
pub mod processor;
pub mod prelude; 
