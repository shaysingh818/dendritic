use std::fs;
use std::fs::File; 
use std::io::{Write, BufWriter, BufReader}; 

use ndarray::Array2;
use indicatif::{ProgressBar, ProgressStyle}; 
use serde::{Serialize, Deserialize}; 

use dendritic_autodiff::operations::arithmetic::*; 
use dendritic_autodiff::operations::loss::*;
use dendritic_autodiff::graph::{ComputationGraph, GraphConstruction, GraphSerialize};

use crate::descent::DescentOptimizer; 

pub struct LinearRegression {

    /// Underlying computation graph with operations for optimizer
    pub graph: ComputationGraph<Array2<f64>>,

    /// Dataset containing features (variables)
    pub inputs: Option<Array2<f64>>,

    /// Dataset containing expected predicted values
    pub outputs: Option<Array2<f64>>,

    /// Coefficients associated with each feature
    pub weights: Array2<f64>,

    /// Bias to add after weights multiplication
    pub bias: Array2<f64>,

    /// Amount of iterations for training cycle
    pub learning_rate: f64

}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionSerialize {
    
    /// Path for where linear regression computation graph is stored
    pub graph_path: String, 

    /// Saved weights for regression model
    pub weights: Array2<f64>,

    /// Saved bias for regression model
    pub bias: Array2<f64>,

    /// Learning rate saved from previous training
    pub learning_rate: f64
}


impl LinearRegression {

    pub fn new(
        x: &Array2<f64>, 
        y: &Array2<f64>, 
        learning_rate: f64) -> Result<LinearRegression, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err("Learning rate must be between 0 and 1".to_string());
        }

        Ok(Self {
            graph: ComputationGraph::new(),
            inputs: Some(x.clone()),
            outputs: Some(y.clone()),
            weights: Array2::zeros((x.shape()[1], 1)),
            bias: Array2::zeros((1, 1)),
            learning_rate: learning_rate
        })

    }

    pub fn predict(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.graph.mut_node_output(0, input.clone()); 
        self.graph.forward();
        self.graph.node(4).output()
    }

}


impl DescentOptimizer for LinearRegression {

    fn function_definition(&mut self) {

        if let Some(inputs) = self.inputs.as_ref() {
            self.graph.mul(vec![inputs.clone(), self.weights.clone()]); 
        } else {
            panic!("Inputs not set for linear regression model");
        }

        self.graph.add(vec![self.bias.clone()]); 

        if let Some(outputs) = self.outputs.as_ref() {
            self.graph.mse(outputs.clone());
        } else {
            panic!("Outputs not set for linear regression model"); 
        }

        self.graph.add_parameter(1); 
        self.graph.add_parameter(3); 

    }

    fn parameter_update(&mut self) {

        if let Some(outputs) = self.outputs.as_ref() {
            for var_idx in self.graph.parameters() {
                let var = self.graph.node(var_idx);
                let grad = var.grad() * (self.learning_rate / outputs.len() as f64);
                let delta = var.output() - grad;
                self.graph.mut_node_output(var_idx, delta.clone());
            }
        }


    }

    fn train(&mut self, epochs: usize) {

        self.function_definition();

        let bar = ProgressBar::new(epochs.try_into().unwrap());
        bar.set_style(ProgressStyle::default_bar()
            .template("{bar:50} {pos}/{len}")
            .unwrap());

        for _ in 0..epochs {
            self.graph.forward();
            self.graph.backward(); 
            self.parameter_update();
            bar.inc(1); 
        }

        bar.finish();

        let loss_node = self.graph.curr_node();
        let loss = loss_node.output();
        println!(
            "Loss: {:?}, Learning Rate: {:?}", 
            loss.as_slice().unwrap()[0],
            self.learning_rate
        );

    }

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = LinearRegressionSerialize {
            graph_path: format!("{filepath}/linear_regression_exp"),
            weights: self.weights.clone(), 
            bias: self.bias.clone(),
            learning_rate: self.learning_rate
        };

        self.graph.save(&obj.graph_path); 

        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file); 
        let json_string = serde_json::to_string_pretty(&obj)?;
        writer.write_all(json_string.as_bytes())?; 
        Ok(())
    }

    fn save_snapshot(&self, namespace: &str) -> std::io::Result<()> {
        Ok(())
    }

    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {

        let parameter_path = format!("{filepath}/parameters.json");
        let obj: LinearRegressionSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(LinearRegression {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            inputs: None, 
            outputs: None, 
            weights: obj.weights, 
            bias: obj.bias,
            learning_rate: obj.learning_rate
        }) 
    }

    fn load_snapshot(namespace: &str) -> Self {
        unimplemented!();
    }

}



