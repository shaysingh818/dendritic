pub mod ndarray;
pub mod regression;
pub mod loss;
pub mod models;

// use crate::models::gtd::*;

use crate::regression::logistic::Logistic;
use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*;

fn main()  {

    let x: NDArray<f64> = NDArray::load("data/logistic_testing_data/inputs").unwrap();
    let y: NDArray<f64> = NDArray::load("data/logistic_testing_data/outputs").unwrap();

    let mut loaded_model = Logistic::load("gtd_iteration1", x, y, 0.01).unwrap();
    let results = loaded_model.forward().unwrap();

    println!("{:?}", loaded_model.outputs().values());
    println!("{:?}", results.values()); 


}
