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


pub struct LassoRegression {

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
    pub learning_rate: f64,

    /// Lambda parameter for regularization of weights
    pub lambda: f64

}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LassoRegressionSerialize {
    
    /// Path for where linear regression computation graph is stored
    pub graph_path: String, 

    /// Saved weights for regression model
    pub weights: Array2<f64>,

    /// Saved bias for regression model
    pub bias: Array2<f64>,

    /// Learning rate saved from previous training
    pub learning_rate: f64,

    /// Lambda parameter for regularization of weights
    pub lambda: f64
}


impl LassoRegression {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64,
        lambda: f64) -> Result<LassoRegression, String> {
 
        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }
 
        Ok(Self {
            graph: ComputationGraph::new(),
            inputs: Some(x.clone()),
            outputs: Some(y.clone()),
            weights: Array2::zeros((x.shape()[1], 1)),
            bias: Array2::zeros((1, 1)),
            learning_rate: learning_rate,
            lambda: lambda
        })
    }

}


impl DescentOptimizer for LassoRegression {

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

            let w = self.graph.node(1);
            let sig_w = self.graph.node(1).output().mapv(|x| x.signum());
            let w_grad = w.clone().grad() + (self.lambda * sig_w);
            let w_new = w.output() - (w_grad * self.learning_rate);
            self.graph.mut_node_output(1, w_new); 

            let b = self.graph.node(3);
            let b_grad = b.grad() * (self.learning_rate / outputs.len() as f64);
            let b_delta = b.output() - b_grad;
            self.graph.mut_node_output(3, b_delta); 

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

        self.weights = self.graph.node(1).output();
        self.bias = self.graph.node(3).output(); 
        let l1_penalty = self.weights.mapv(|x| x.abs()).sum();

        let loss_node = self.graph.curr_node();
        let loss = loss_node.output();
        let total_loss = loss.clone() + (self.lambda * l1_penalty);
        println!(
            "Loss: {:?}, Learning Rate: {:?}, Lambda: {:?}", 
            loss.as_slice().unwrap()[0],
            self.learning_rate,
            self.lambda
        ); 

    }

    fn train_batch(&mut self, epochs: usize, batch_size: usize) {

        if epochs % 1000 != 0 {
            panic!("Number of epochs must be evenly divisble by 1000");
        }

        self.function_definition();

        let inputs = self.inputs.as_ref().expect("Inputs not defined");
        let outputs = self.outputs.as_ref().expect("Outputs not defined");
        let x_train = inputs.clone();
        let y_train = outputs.clone(); 
        let rows = x_train.nrows();
        let num_batches = (rows + batch_size - 1) / batch_size;

        let epoch_batches = epochs / 1000;
        for _ in 0..epoch_batches {
            
            let bar = ProgressBar::new(1000);
            bar.set_style(ProgressStyle::default_bar()
                .template("{bar:50} {pos}/{len}")
                .unwrap());

            for epoch_idx in 0..1000 {

                let mut row_indices: Vec<_> = (0..rows).collect();
                row_indices.shuffle(&mut thread_rng());

                let x_shuffled = x_train.select(Axis(0), &row_indices);
                let y_shuffled = y_train.select(Axis(0), &row_indices);

                for batch_idx in 0..num_batches { 
                    let start_idx = batch_idx * batch_size;
                    let end_idx = (start_idx + batch_size).min(rows);
                    let x = x_shuffled.slice(s![start_idx..end_idx, ..]);
                    let y = y_shuffled.slice(s![start_idx..end_idx, ..]);

                    // fix this later
                    if (end_idx - start_idx) < batch_size {
                        continue; 
                    }

                    self.graph.mut_node_output(0, x.to_owned());
                    self.graph.mut_node_output(4, y.to_owned()); 
                    self.graph.mut_node_output(5, y.to_owned());

                    self.graph.forward();
                    self.graph.backward(); 
                    self.parameter_update();
                }
                bar.inc(1); 
            }

            bar.finish();

            self.weights = self.graph.node(1).output();
            self.bias = self.graph.node(3).output(); 
            let l1_penalty = self.weights.mapv(|x| x.abs()).sum();

            let loss_node = self.graph.curr_node();
            let loss = loss_node.output();
            let total_loss = loss.clone() + (self.lambda * l1_penalty);
            println!(
                "Loss: {:?}, Learning Rate: {:?}, Lambda: {:?}", 
                loss.as_slice().unwrap()[0],
                self.learning_rate,
                self.lambda
            ); 
            println!(""); 

        }
    }

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = LassoRegressionSerialize {
            graph_path: format!("{filepath}/lasso_regression_exp"),
            weights: self.weights.clone(), 
            bias: self.bias.clone(),
            learning_rate: self.learning_rate,
            lambda: self.lambda
        };

        let _ = self.graph.save(&obj.graph_path); 

        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file); 
        let json_string = serde_json::to_string_pretty(&obj)?;
        writer.write_all(json_string.as_bytes())?; 
        Ok(())
    }

    fn save_snapshot(&self, namespace: &str) -> std::io::Result<()> {

        let now = Utc::now();
        let (_, year) = now.year_ce();
        let month = now.month().to_string();
        let day = now.day().to_string();
        let curr_year = year.to_string();

        let directory_path = format!("{namespace}/snapshot/{curr_year}/{month}/{day}");
        fs::create_dir_all(directory_path.clone())?; 

        let id = Uuid::new_v4();
        let file_path = format!("{directory_path}/{id}.json");

        let obj = LassoRegressionSerialize {
            graph_path: format!("{namespace}/lasso_regression_exp"),
            weights: self.weights.clone(), 
            bias: self.bias.clone(),
            learning_rate: self.learning_rate,
            lambda: self.lambda
        };

        let _ = self.graph.save(&obj.graph_path); 

        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file); 
        let json_string = serde_json::to_string_pretty(&obj)?;
        writer.write_all(json_string.as_bytes())?; 
        Ok(())
    }

    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {

        let parameter_path = format!("{filepath}/parameters.json");
        let obj: LassoRegressionSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(LassoRegression {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            inputs: None, 
            outputs: None, 
            weights: obj.weights, 
            bias: obj.bias,
            learning_rate: obj.learning_rate,
            lambda: obj.lambda
        }) 

    }

    fn load_snapshot(
        namespace: &str,
        year: &str,
        month: &str,
        day: &str,
        snapshot_id: &str) -> Result<Self, Box<dyn std::error::Error>> {

        let parameter_path = format!(
            "{namespace}/snapshot/{year}/{month}/{day}/{snapshot_id}.json"
        );

        let obj: LassoRegressionSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(LassoRegression {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            inputs: None, 
            outputs: None, 
            weights: obj.weights, 
            bias: obj.bias,
            learning_rate: obj.learning_rate,
            lambda: obj.lambda
        }) 

    }


}
