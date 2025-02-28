use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;
use dendritic_metrics::loss::*;
use ndarray::prelude::*; 
use ndarray::{Array2}; 
use std::fs;


#[derive(Debug)]
pub struct LinearSolver {
    pub X: Array2<f64>,
    pub b: Array2<f64>,
    pub parameters: Array2<f64>,
    pub iterations: usize,
    pub threshold: f64
}


impl LinearSolver {

    pub fn new(
        X: &Array2<f64>,
        b: &Array2<f64>, 
        threshold: f64) -> Result<LinearSolver, String> {

        if X.dim().0 != X.dim().1 {
            let msg = format!(
                "Input must be a square matrix {:?} != {:?}",
                X.dim().0, X.dim().1
            );
            return Err(msg.to_string());
        }

        if b.dim().1 > 1 {
            let msg = format!(
                "Target vector can't have more than 1 column: {:?}",
                b.dim().1
            ); 
            return Err(msg.to_string()); 
        }

        if b.dim().0 != X.dim().0 {
            let msg = format!(
                "Solution set rows not equal {:?} != {:?}",
                X.dim().0, b.dim().0
            ); 
            return Err(msg.to_string()); 
        }

        Ok(Self {
            X: X.clone(),
            b: b.clone(),
            iterations: 0,
            threshold: threshold,
            parameters: Array2::zeros((X.dim().1, 1)),
        })

    }


    pub fn gauss_seidal(&mut self) -> Result<(), Box<dyn std::error::Error>> {

        let n = self.X.dim().0;
        let mut loss: f64 = f64::INFINITY;

        while true {

            let current_params = self.parameters.clone();

            for i in 0..n {
                let mut theta: f64 = 0.0;
                for j in 0..n {
                    if j != i {
                        theta += self.X[[i, j]] * self.parameters[[j, 0]];
                    }
                }
                let val = (self.b[[i, 0]] - theta) / self.X[[i, i]]; 
                self.parameters[[i, 0]] = val; 
            }

            let new_params = self.parameters.clone();
            let diff = current_params - new_params;
            if diff.sum() == self.threshold {
                break
            }

            self.iterations += 1;
        } 

        Ok(())
    }

    pub fn sor(&mut self, w: f64) -> Result<(), Box<dyn std::error::Error>> {

        let n = self.X.dim().0;
        let mut loss: f64 = f64::INFINITY;

        while true {

            let current_params = self.parameters.clone();

            for i in 0..n {
                let mut theta: f64 = 0.0;
                for j in 0..n {
                    if j != i {
                        theta += self.X[[i, j]] * self.parameters[[j, 0]];
                    }
                }
                
                let p1 = (1.0 - w) * self.parameters[[i, 0]];
                let p2 = (w / self.X[[i, i]]) * (self.b[[i, 0]] - theta);
                let val = p1 + p2; 
                self.parameters[[i, 0]] = val; 
            }

            let new_params = self.parameters.clone();
            let diff = current_params - new_params;
            if diff.sum() <= self.threshold {
                break; 
            }

            self.iterations += 1;
        } 

        Ok(())
    }

}

