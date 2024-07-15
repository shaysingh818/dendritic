# Dendrite
Dendrite is a general purpose supervised/un-supervised machine learning library written for the rust ecosystem. It contains the required data structures & algorithms needed for general machine learning. It acts as core library with packages for predictive data modeling.

# Table of Contents

* [Examples](#Examples)
* [NDArray](#NDArray)
* [Autodiff](#Autodiff)
* [Metrics](#Metrics)
* [Regression](#Regression)
* [Data sets](#Data%20sets)


## Examples
* Examples of creating predictive models using the Dedrite package
* Current examples are classification based with logistic regression
* Demonstrates preprocessing capbilities along with metrics functions for measuring loss/accuracy


## IRIS Flowers Prediction

```rust
use datasets::iris::*;
use regression::logistic::*;
use metrics::loss::*;
use metrics::activations::*;
use preprocessing::encoding::*;

// load data
let (x_train, y_train) = load_iris().unwrap();

// encode the target variables
let mut encoder = OneHotEncoding::new(y_train.clone()).unwrap();
let y_train_encoded = encoder.transform();

// create logistic regression model
let mut log_model = MultiClassLogistic::new(
    x_train.clone(),
    y_train_encoded.clone(),
    softmax,
    0.1
).unwrap();

log_model.sgd(500, true, 5);

let sample_index = 50;
let x_test = x_train.batch(5).unwrap();
let y_test = y_train.batch(5).unwrap();
let y_pred = log_model.predict(x_test[sample_index].clone());

println!("Actual: {:?}", y_test[sample_index]);
println!("Prediction: {:?}", y_pred.values());

let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
println!("LOSS: {:?}", loss);  
```





