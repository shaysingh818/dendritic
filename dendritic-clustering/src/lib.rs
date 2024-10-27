//! # Dendritic Clustering Crate
//!
//! This crate allows for clustering of data for unsupervised tasks.
//! The bayes crate currently supports K means clustering.
//! Code for the Hierarchical clustering module is there but does not work at the moment
//!
//! ## Features
//! - **K Means**: Standard K means clustering
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-clustering = "0.1"
//! ```
//!
//! ## Example Usage
//! This is an example of using the K means clustering module
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_clustering::k_means::*;
//! use dendritic_knn::distance::*;
//! use dendritic_datasets::iris::*; 
//!
//! fn main() {
//!
//!     // Load datasets from saved ndarray
//!     let data_path = "../dendritic-datasets/data/iris.parquet";
//!     let (x_train, y_train) = load_iris(data_path).unwrap();
//!
//!     // Iterations and K value for K means cluster
//!     let iterations = 5;
//!     let k_value = 3; 
//!
//!     // Create instance of K means model
//!     let mut clf = KMeans::new(
//!         &x_train, 
//!         k_value, 
//!         iterations, 
//!         euclidean
//!     ).unwrap();
//!
//!     // Get centroids
//!     let final_centroids = clf.fit();
//!     let centroids_unique = final_centroids.unique();
//! }
//! ```
pub mod k_means;
pub mod hierarchical;
