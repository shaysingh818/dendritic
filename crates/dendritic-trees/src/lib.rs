//! # Dendritic Trees Crate
//!
//! This crate contains all tree based machine learning models.
//! Contains standard decision tree and random forest classifier and regressors.
//!
//! ## Features
//! - **Decision Tree**: Standard scalar and min max normlization of data.
//! - **Random Forest**: One hot encoding for multi class data
//! - **Bootstrap**: One hot encoding for multi class data
//!
//! ## Getting Started
//! To get started, add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! dendritic-trees = "1.1.0"
//! ```
//! ## Example Usage
//! This is an example of using the decision tree classifier model provided by dendritic.
//! The example below uses decision trees but random forest can be used with  `RandomForestClassifier` or `RandomForestRegressor`.
//! ```rust
//! use dendritic_ndarray::ndarray::NDArray;
//! use dendritic_ndarray::ops::*;
//! use dendritic_metrics::loss::*;
//! use dendritic_metrics::utils::*;
//! use dendritic_trees::decision_tree::*;
//! use dendritic_datasets::iris::*;
//! 
//! fn main() {
//!
//!    // load data
//!    let data_path = "../dendritic-datasets/data/iris.parquet";
//!    let max_depth = 3;
//!    let samples_split = 3; 
//!    let (x_train_test, y_train_test) = load_iris(data_path).unwrap();
//!    let (x_train, y_train) = load_all_iris(data_path).unwrap();
//!    
//!    // Decision tree classifier model
//!    let mut model = DecisionTreeClassifier::new(
//!        max_depth, 
//!        samples_split, 
//!        gini_impurity
//!    );
//!    model.fit(&x_train, &y_train);
//!
//!    let sample_index = 100;
//!    let x_test = x_train_test.batch(5).unwrap();
//!    let y_test = y_train_test.batch(5).unwrap();
//!    let y_pred = model.predict(x_test[sample_index].clone());
//!    println!("Actual: {:?}", y_test[sample_index]);
//!    println!("Prediction: {:?}", y_pred.values()); 
//!
//! }
//! ```
//! ## Disclaimer
//! The dendritic project is a toy machine learning library built for learning and research purposes.
//! It is not advised by the maintainer to use this library as a production ready machine learning library.
//! This is a project that is still very much a work in progress.
pub mod decision_tree_regressor;
pub mod decision_tree;
pub mod random_forest;
pub mod node;
pub mod utils;
pub mod bootstrap; 
