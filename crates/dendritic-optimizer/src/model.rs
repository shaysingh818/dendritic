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


pub trait Model {

    /// Inputs fed to model
    fn input(&self) -> Array2<f64>;

    /// True labels of prediction
    fn output(&self) -> Array2<f64>; 

    /// Set inputs for model
    fn set_input(&mut self, x: &Array2<f64>);

    /// Set true labels for model
    fn set_output(&mut self, y: &Array2<f64>); 

    /// Predicted output based on parameters of model
    fn predicted(&self) -> Array2<f64>;

    /// Measure loss for a model
    fn loss(&mut self) -> f64;

    /// Update paramters for a model
    fn update_parameters(&mut self);

}

pub trait ModelSerialize {

    /// Save routine for saving single instance of parameters
    fn save(&self, filepath: &str) -> std::io::Result<()>;

    /// Save routine for saving snapshots of parameters at specific time intervals
    fn save_snapshot(&self, namespace: &str) -> std::io::Result<()>;
 
    /// Load routine for loading single instance of parameters
    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized;
 
    /// Load routine for loading snapshot of parameters for a specific date
    fn load_snapshot(
        namespace: &str,
        year: &str,
        month: &str,
        day: &str,
        snapshot_id: &str) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized;

}


pub trait Optimizer {

    /// Retrieve parameter indexes to update for optimizer
    fn parameters(&self) -> Vec<usize>;

    /// Step for updating parameters of a given optimizer
    fn step(&mut self); 

    /// Reset gradients of parameters back to 0
    fn reset(&mut self);
}


/// Traits for creating model pipelines with the library
pub trait ModelPipeline {
    
    /// Instantiate new model
    fn register(name: &str) -> Self;

}

pub trait Load: ModelPipeline {

    /// Load data to model
    fn load(&self);
}

pub trait Transform: ModelPipeline {

    /// Transform data to numerical format (if required)
    fn transform(&self);
}

pub trait Train: ModelPipeline {

    /// Train model on transformed data
    fn train(&self);
}

pub trait Inference: ModelPipeline {

    /// Predict data on trained model
    fn inference(&self);
}
