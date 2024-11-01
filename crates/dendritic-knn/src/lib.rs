//! # Dendritic K Nearest Neighbors Crate
//!
//! This crate contains functionality for performing K nearest neighbors for classification and regression.
//! Package also contains all distance metrics that can be used across dendritic.
//!
//! ## Features
//! - **KNN**: KNN regressor and classifier.
//! - **Distance**: Module with various distance metrics
//!
//! ## Example Usage
//! This is an example of using the KNN classifier
//! ```rust
//! use dendritic_datasets::iris::*;
//! use dendritic_knn::knn::*;
//! use dendritic_knn::distance::*; 
//!
//! fn main() {
//!
//!    // Load iris flowers dataset
//!    let (x, y) = load_iris("../dendritic-datasets/data/iris.parquet").unwrap();
//!    let (x_train, x_test) = x.split(0, 0.80).unwrap(); // split rows with 80/20 split
//!    let (y_train, y_test) = y.split(0, 0.80).unwrap();
//!
//!    let clf = KNN::fit(
//!        &x_train, 
//!        &y_train, 
//!        4, 
//!        euclidean
//!    ).unwrap();
//!
//!    let predictions = clf.predict(&x_test);
//!    println!("Actual: {:?}", predictions.values());
//!    println!("Prediction: {:?}", y_test.values()); 
//!
//! }
//! ```
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod distance;
pub mod knn;
pub mod utils;  
