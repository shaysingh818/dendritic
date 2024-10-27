//! # Dendritic Bayesian Statistics Crate
//!
//! This crate allows for common bayesian methods for regression and classification tasks.
//! The bayes crate currently supports guassian and standard naive bayes.
//!
//! ## Features
//! - **Guassian Bayes**: Bayesian model that uses gaussian density function for predicting likelihoods
//! - **Naive Bayes**: Standard naive bayes model
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-bayes = "0.1"
//! ```
//!
//! ## Example Usage
//! This is an example of using both the naive and gaussian bayes models
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_bayes::naive_bayes::*;
//! use dendritic_bayes::gaussian_bayes::*;

//!
//! fn main() {
//!     // Load datasets from saved ndarray
//!     let x_path = "data/weather_multi_feature/inputs";
//!     let y_path = "data/weather_multi_feature/outputs";
//!
//!     // Load saved ndarrays in memory
//!     let features = NDArray::load(x_path).unwrap();
//!     let target = NDArray::load(y_path).unwrap();
//!
//!     // Create instance of naive bayes model
//!     let mut nb_clf = NaiveBayes::new(
//!         &features,
//!         &target
//!     ).unwrap();
//!
//!     // Create instance of guassian bayes model
//!     let mut gb_clf = GaussianNB::new(
//!         &features,
//!         &target
//!     ).unwrap();
//!     
//!     // Make prediction with first row of features
//!     let row1 = features.axis(0, 0).unwrap();
//!     let nb_pred = nb_clf.fit(row1);
//!     let gb_pred = gb_clf.fit(row1.clone()); // This will take in references eventually
//! }
//! ```
pub mod shared; 
pub mod naive_bayes;
pub mod gaussian_bayes; 
