
//! # Dendritic Datasets Crate
//!
//! This crate contains all the preloaded datasets for dendritic.
//! Most of these datasets come from kaggle and have been converted to parquet files to work with the apache arrow crate.
//! Dendritic does not support any known dataframe libraries at the moment. 
//! Dendritic works with anything that can be converted to it `NDArray<f64>` data structure. 
//!
//! ## Datasets
//! - **Diabetes**: Diabetes dataset for binary classification tasks.
//! - **Iris**: Iris flowers dataset for multi class classification tasks
//! - **Breast Cancer**: Breast Cancer diagnosis for binary classification
//! - **Alzhiemers**: Alzhiemers diagnosis data amongst adults
//! - **Customer Purchase**: Customer purchase data for multi class classification
//! - **Boston Housing**: Boston housing data for regression tasks
//! - **Student Performance**: Student test scores for regression tasks
//! - **Airfoil Nooise**: Airfoil noise data for regression tasks
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-datasets = "0.1"
//! ```
//!
//! ## Example Usage
//! This is an example of using any of the preloaded datasets for dendritic
//! ```rust
//! use dendritic_datasets::iris::*;
//! use dendritic_datasets::breast_cancer::*;
//! use dendritic_datasets::diabetes::*;
//! use dendritic_datasets::alzhiemers::*;
//! use dendritic_datasets::customer_purchase::*;
//! use dendritic_datasets::student_performance::*;
//! use dendritic_datasets::airfoil_noise::*;
//! 
//! fn main() {
//!
//!     // Examples of loading the datasets
//!    let diabetes = "../dendritic-datasets/data/diabetes.parquet";
//!    let (x_train, y_train) = load_diabetes(diabetes).unwrap();
//!
//!    let breast_cancer = "../dendritic-datasets/data/breast_cancer.parquet";
//!    let (x_train, y_train) = load_breast_cancer(breast_cancer).unwrap();
//!
//!    let iris_data = "../dendritic-datasets/data/iris.parquet";
//!    let (x_train, y_train) = load_iris(iris_data).unwrap();
//!
//!    let alz = "../dendritic-datasets/data/alzheimers.parquet";
//!    let (x_train, y_train) = load_alzhiemers(alz).unwrap();
//!
//!    // Refer to crate docs for loading other datasets
//! }
//! ```
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod utils;
pub mod diabetes;
pub mod iris; 
pub mod breast_cancer;
pub mod alzhiemers;
pub mod customer_purchase;
pub mod student_performance; 
pub mod boston_housing;
pub mod airfoil_noise;
