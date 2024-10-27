

//! # Dendritic NDArray Crate
//!
//! This crate is the numerical computing crate to work with N Dimensional values.
//! The operations for this numerical computing library are broken down by aggregate, binary, scalar and unary operations.
//! This crate serves as the base depedencies for most numerical computing for all the machine learning algorithms dendritic has to offer
//!
//! ## Features
//! - **NDArray**: General NDArray structure that can work with generic values.
//! - **Shape**: Shape structure for representing dimension of N Dimensional value
//! - **Ops**: Operations broken down into different categories supported by the NDArray module
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-ndarray = "0.1"
//! ```
//!
//! ## Supported operation types
//! Currently the numerical operations are only supported for `f64` types. This will change in future releases.

//! ## Binary Operations Example Usage
//! These are some examples of using the `dendritic_ndarray` create with some basic operations
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*; 
//!
//! fn main() {
//!     // Create 2 instance of ndarrays with a shape of (2,3) for testing binary operations
//!     let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
//!     let y: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
//!     
//!     // perform add operation
//!     let add_result : NDArray<f64> = x.add(y).unwrap();
//!     // save result to json file
//!     add_result.save("name_of_saved_ndarray").unwrap();
//!     // load result back to new ndarray
//!     let loaded = NDArray::load("name_of_saved_ndarray").unwrap();
//! }
//! ```


//! ## Unary Operations Example Usage
//! These are some examples of using the `dendritic_ndarray` with the unary operations. 
//! Unary operations involve transoforming or modifying the contents of an ndarray and then returning the transformed result. 
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*; 
//!
//! fn main() {
//!     // Create instance of ndarray, multiply by f64 scalar value
//!     let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
//!     
//!     // perform transpose operation
//!     let y : NDArray<f64> = x.transpose().unwrap(); // will return result with shape (3, 2)
//! }
//! ```


//! ## Scalar Operations Example Usage
//! These are some examples of using the `dendritic_ndarray` with the scalar operations. 
//! Scalar operations involve an `ndarray` and a scalar value like an `f64`. 
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*; 
//!
//! fn main() {
//!     // Create instance of ndarray, multiply by f64 scalar value
//!     let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
//!     let y: f64 = 10.0;
//!     
//!     // perform add operation
//!     let scalar_result : NDArray<f64> = x.scalar_add(y).unwrap();
//! }
//! ```


//! ## Aggregate Operations Example Usage
//! These are some examples of using the `dendritic_ndarray` with the aggregate operations. 
//! Aggregate operations reduce the dimension or result of an operation on an ndarray. 
//! An example of this is statistical operations like taking the average or summing all values.  
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*; 
//!
//! fn main() {
//!     // Create instance of ndarray, multiply by f64 scalar value
//!     let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
//!     
//!     // perform add operation
//!     let x_avg: f64 = x.avg();
//! }
//! ```
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod ndarray;
pub mod shape; 
pub mod ops;



