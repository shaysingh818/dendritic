use std::fs;
use std::fs::File; 
use std::io::{Write, BufWriter, BufReader}; 

use rand::thread_rng;
use rand::prelude::SliceRandom;
use uuid::Uuid;
use chrono::{Datelike, Utc};  
use ndarray::{s, Array2, Axis};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress}; 
use serde::{Serialize, Deserialize}; 

use dendritic_autodiff::operations::arithmetic::*; 
use dendritic_autodiff::operations::loss::*;
use dendritic_autodiff::graph::{ComputationGraph, GraphConstruction, GraphSerialize};

use crate::descent::DescentOptimizer; 


pub struct Regression {

    /// Underlying computation graph with operations for optimizer
    pub graph: ComputationGraph<Array2<f64>>,

    /// Dataset containing features (variables)
    pub inputs: Option<Array2<f64>>,

    /// Dataset containing expected predicted values
    pub outputs: Option<Array2<f64>>,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Amount of iterations for training cycle
    pub learning_rate: f64,

    /// Lambda parameter for regularization of weights
    pub lambda: f64,

    /// Alpha mixing rate for elastic net (l1 & l2) regularization
    pub alpha f64

}


impl Regression {

    pub fn linear(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64) -> Result<Regression, String> {
 
        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }

        Ok(Self {
            graph: ComputationGraph::new(),
            inputs: Some(x.clone()),
            outputs: Some(y.clone()),
            weights: (x.shape()[1], 1),
            bias: (1, 1),
            learning_rate: learning_rate,
            lambda: 0.00,
            alpha: 0.00
        })
    }

    pub fn ridge(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64,
        lambda: f64) -> Result<Regression, String> {
 
        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }

        Ok(Self {
            graph: ComputationGraph::new(),
            inputs: Some(x.clone()),
            outputs: Some(y.clone()),
            weights: (x.shape()[1], 1),
            bias: (1, 1),
            learning_rate: learning_rate,
            lambda: lambda,
            alpha: 0.00
        })
    }

    pub fn lasso(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64,
        lambda: f64) -> Result<Regression, String> {
 
        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }

        Ok(Self {
            graph: ComputationGraph::new(),
            inputs: Some(x.clone()),
            outputs: Some(y.clone()),
            weights: (x.shape()[1], 1),
            bias: (1, 1),
            learning_rate: learning_rate,
            lambda: lambda,
            alpha: 0.00
        })
    }

    fn function_definition(&mut self) {

        if let Some(inputs) = self.inputs.as_ref() {
            let w = Array2::zeros(self.weight_dim); 
            self.graph.mul(vec![inputs.clone(), w]); 
        } else {
            panic!("Inputs not set for regression model");
        }

        self.graph.add(vec![Array2::zeros(self.bias_dim)]); 

        if let Some(outputs) = self.outputs.as_ref() {
            self.graph.mse(outputs.clone());
        } else {
            panic!("Outputs not set for regression model"); 
        }

        self.graph.add_parameter(1); 
        self.graph.add_parameter(3); 

    }

}


pub trait Regression {

    fn parameter_update(&mut self);

    fn measure_loss(&mut self);

    fn save(&self, filepath: &str) -> std::io::Result<()>;

    fn save_snapshot(&self, namespace: &str) -> std::io::Result<()>;
 
    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>>;
 
    fn load_snapshot(
        namespace: &str,
        year: &str,
        month: &str,
        day: &str,
        snapshot_id: &str) -> Result<Self, Box<dyn std::error::Error>>;

}


