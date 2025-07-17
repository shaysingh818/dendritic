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


use dendritic_autodiff::operations::base::*; 
use dendritic_autodiff::operations::arithmetic::*; 
use dendritic_autodiff::operations::loss::*;
use dendritic_autodiff::operations::activation::*;
use dendritic_autodiff::graph::{ComputationGraph, GraphConstruction, GraphSerialize};
use crate::regression::*; 


pub struct Logistic {

    /// Underlying computation graph with operations for optimizer
    pub graph: ComputationGraph<Array2<f64>>,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Learning rate to control how fast to decrease
    pub learning_rate: f64
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticSerialize {

    /// Underlying computation graph with operations for optimizer
    pub graph_path: String,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Learning rate to control how fast to decrease
    pub learning_rate: f64
}


impl Logistic {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64) -> Result<Self, String> {

        let mut regression = Logistic {
            graph: ComputationGraph::new(),
            weight_dim: (x.shape()[1], 1),
            bias_dim: (1, 1),
            learning_rate: learning_rate
        };

        regression.graph.mul(
            vec![
                x.clone(), 
                Array2::zeros(regression.weight_dim)
            ]
        );

        regression.graph.add(
            vec![Array2::zeros(regression.bias_dim)]
        );

        regression.graph.sigmoid();
        regression.graph.bce(y.clone()); 

        regression.graph.add_parameter(1);
        regression.graph.add_parameter(3);
        Ok(regression)
    }

    pub fn input(&self) -> Array2<f64> {
        self.graph.node(0).output()
    }

    pub fn output(&self) -> Array2<f64> {
        self.graph.node(6).output()
    }

    pub fn predicted_output(&self) -> Array2<f64> {
        self.graph.node(5).output()
    }

    pub fn set_input(&mut self, x: &Array2<f64>) {
        self.graph.mut_node_output(0, x.to_owned());
    }

    pub fn set_output(&mut self, y: &Array2<f64>) {
        self.graph.mut_node_output(6, y.to_owned());
        self.graph.mut_node_output(7, y.to_owned());
    }

    pub fn set_loss(&mut self, op: Box<dyn Operation<Array2<f64>>>) {
        self.graph.mut_node_operation(7, op); 
    }

    pub fn set_activation(&mut self, op: Box<dyn Operation<Array2<f64>>>) {
        self.graph.mut_node_operation(5, op); 
    }

}


impl RegressionOptimizer for Logistic {

    fn parameter_update(&mut self) {

        let w = self.graph.node(1);
        let w_grad = w.grad() * self.learning_rate;
        let w_delta = w.output() - w_grad;
        self.graph.mut_node_output(1, w_delta); 

        let b = self.graph.node(3);
        let b_grad = (b.grad() * self.learning_rate).sum_axis(Axis(0));
        let b_delta = b.output() - b_grad;
        self.graph.mut_node_output(3, b_delta); 
    }

    fn measure_loss(&mut self) -> f64 {
        
        let loss_node = self.graph.curr_node();
        let loss = loss_node.output();
        loss.as_slice().unwrap()[0]
    }

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = LogisticSerialize {
            graph_path: format!("{filepath}/regression_exp"),
            weight_dim: self.weight_dim,
            bias_dim: self.bias_dim,
            learning_rate: self.learning_rate
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


        let obj = LogisticSerialize {
            graph_path: format!("{namespace}/regression_exp"),
            weight_dim: self.weight_dim,
            bias_dim: self.bias_dim,
            learning_rate: self.learning_rate
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
        let obj: LogisticSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(Logistic {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            weight_dim: obj.weight_dim,
            bias_dim: obj.bias_dim,
            learning_rate: obj.learning_rate
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

        let obj: LogisticSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(Logistic {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            weight_dim: obj.weight_dim,
            bias_dim: obj.bias_dim,
            learning_rate: obj.learning_rate
        })
    }

}
