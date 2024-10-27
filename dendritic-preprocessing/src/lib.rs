
//! # Dendritic Preprocessing Crate
//!
//! This crate contains functionality for performing normalization of data during the preprocessing stage for a model.
//! Contains preprocessing for encoding and standard scaling.
//!
//! ## Features
//! - **Standard Scalar**: Standard scalar and min max normlization of data.
//! - **Encoding**: One hot encoding for multi class data
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-preprocessing = "0.1"
//! ```
//! ## Example Usage
//! This is an example of using the one hot encoder for data with multiple class labels
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_preprocessing::encoding::{OneHotEncoding};
//! 
//! fn main() {
//!
//!    // Data to one hot encode for multi class classification
//!    let x = NDArray::array(vec![10, 1], vec![
//!        1.0,2.0,0.0,2.0,0.0,
//!        0.0,1.0,0.0,2.0,2.0
//!    ]).unwrap();
//!
//!    let mut encoder = OneHotEncoding::new(x).unwrap();
//!    println!("Max Value: {:?}", encoder.max_value()); // 3.0
//!    println!("Num Samples: {:?}", encoder.num_samples()); // 10.0 
//!
//!    let encoded_vals = encoder.transform();
//!    println!("Encoded Values: {:?}", encoded_vals()); 
//!
//! }
//! ```
pub mod standard_scalar;
pub mod encoding; 
