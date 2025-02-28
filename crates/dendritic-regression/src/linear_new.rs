use ndarray::prelude::*; 
use ndarray::{Array2}; 
use ndarray_linalg::Inverse; 
use std::fs; 

#[derive(Debug)]
pub struct LinearRegression {
    pub X: Array2<f64>,
    pub y: Array2<f64>,
    pub coefficients: Array2<f64> 
}

impl LinearRegression {

    pub fn new(
        X: &Array2<f64>, 
        y: &Array2<f64>) -> Result<LinearRegression, String> {

        if X.dim().0 != y.dim().0 {
            let err_msg = format!(
                "Rows of X and y must be equal: {:?} != {:?}",
                X.dim().0, y.dim().0
            ); 
            return Err(err_msg.to_string());
        }

        Ok(Self {
            X: X.clone(),
            y: y.clone(),
            coefficients: Array2::zeros((X.dim().1, 1))
        })

    }

    pub fn fit(&mut self) -> Result<(), String> {

        let lhs = self.X.dot(&self.X.t());
        println!("{:?}", lhs.inv()); 

        Ok(())

    }


}
