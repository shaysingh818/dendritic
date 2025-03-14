 # Dendritic Bayesian Statistics Crate

 This crate allows for common bayesian methods for regression and classification tasks.
 The bayes crate currently supports guassian and standard naive bayes.

 ## Features
 - **Guassian Bayes**: Bayesian model that uses gaussian density function for predicting likelihoods
 - **Naive Bayes**: Standard naive bayes model

# Disclaimer
The dendritic project is a toy machine learning library built for learning and research purposes.
It is not advised by the maintainer to use this library as a production ready machine learning library.
This is a project that is still very much a work in progress.

 ## Getting Started
 To get started, add this to your `Cargo.toml`:
 ```toml
 [dependencies]
 dendritic-bayes = "1.1.0"
 ```

 ## Example Usage
 This is an example of using both the naive and gaussian bayes models
 ```rust
 use dendritic_ndarray::ndarray::NDArray;
 use dendritic_ndarray::ops::*;
 use dendritic_bayes::naive_bayes::*;
 use dendritic_bayes::gaussian_bayes::*;


 fn main() {
     // Load datasets from saved ndarray
     let x_path = "data/weather_multi_feature/inputs";
     let y_path = "data/weather_multi_feature/outputs";

     // Load saved ndarrays in memory
     let features = NDArray::load(x_path).unwrap();
     let target = NDArray::load(y_path).unwrap();

     // Create instance of naive bayes model
     let mut nb_clf = NaiveBayes::new(
         &features,
         &target
     ).unwrap();

     // Create instance of guassian bayes model
     let mut gb_clf = GaussianNB::new(
         &features,
         &target
     ).unwrap();
     
     // Make prediction with first row of features
     let row1 = features.axis(0, 0).unwrap();
     let nb_pred = nb_clf.fit(row1.clone());
     let gb_pred = gb_clf.fit(row1.clone()); // This will take in references eventually
 }
 ```