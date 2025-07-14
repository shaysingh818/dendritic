use std::fs;
use std::fs::File; 
use std::io::{Write, BufWriter, BufReader}; 

use rand::thread_rng;
use rand::prelude::SliceRandom;
use uuid::Uuid;
use chrono::{Datelike, Utc};  
use ndarray::{s, Array2, Axis};
use indicatif::{ProgressBar, ProgressStyle}; 
use serde::{Serialize, Deserialize}; 

use dendritic_autodiff::operations::arithmetic::*; 
use dendritic_autodiff::operations::loss::*;
use dendritic_autodiff::graph::{ComputationGraph, GraphConstruction, GraphSerialize};
use crate::regression::*; 


pub struct Logistic {

    /// Extend regression structure for logistic
    regression: Regression
};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticSerialize {

    /// Serializable instance of regression structure
    regression: RegressionSerialize,
}


impl Logistic {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64) -> Result<Self, String> {

        Ok(Self {
            regression: Regression::new(x, y, learning_rate).unwrap()
        })
    }

}
