use std::fs;
use std::fs::File; 
use std::io::{Write, BufWriter, BufReader};
use std::collections::HashMap; 

use rand::thread_rng;
use rand::prelude::SliceRandom;
use uuid::Uuid;
use chrono::{Datelike, Utc};  
use ndarray::{s, Array2, Axis};
use serde::{Serialize, Deserialize}; 

use dendritic_autodiff::operations::arithmetic::*; 
use dendritic_autodiff::operations::loss::*;
use dendritic_autodiff::graph::{ComputationGraph, GraphConstruction, GraphSerialize};

use crate::optimizers::*; 
use crate::model::*; 


pub struct SGD {

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
pub struct SGDSerialize {

    /// Underlying computation graph with operations for optimizer
    pub graph_path: String,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Learning rate to control how fast to decrease
    pub learning_rate: f64
}


impl SGD {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64) -> Result<Self, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }

        let mut sgd = SGD {
            graph: ComputationGraph::new(),
            weight_dim: (x.shape()[1], 1),
            bias_dim: (1, 1),
            learning_rate: learning_rate
        };

        sgd.graph.mul(vec![x.clone(), Array2::zeros(sgd.weight_dim)]); 
        sgd.graph.add(vec![Array2::zeros(sgd.bias_dim)]);
        sgd.graph.mse(y.clone());

        sgd.graph.add_parameter(1);
        sgd.graph.add_parameter(3);
        Ok(sgd)
    }

}


impl Model for SGD {
    
    fn input(&self) -> Array2<f64> {
        self.graph.node(0).output()
    }

    fn output(&self) -> Array2<f64> {
        self.graph.node(5).output()
    }

    fn set_input(&mut self, x: &Array2<f64>) {
        self.graph.mut_node_output(0, x.to_owned());
    }

    fn set_output(&mut self, y: &Array2<f64>) {
        self.graph.mut_node_output(4, y.to_owned());
        self.graph.mut_node_output(5, y.to_owned());
    }

    fn predicted(&self) -> Array2<f64> {
        self.graph.node(4).output()
    }

    fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.set_output(&Array2::zeros((x.nrows(), 1)));
        self.set_input(x);
        self.graph.forward();
        self.predicted()
    }

    fn loss(&mut self) -> f64 {
        let loss_node = self.graph.curr_node();
        let loss = loss_node.output();
        loss.as_slice().unwrap()[0]
    }

    fn update_parameters(&mut self) {

        let w = self.graph.node(1);
        let w_grad = w.grad() * self.learning_rate;
        let w_delta = w.output() - w_grad;
        self.graph.mut_node_output(1, w_delta); 

        let b = self.graph.node(3);
        let b_grad = b.grad() * self.learning_rate;
        let b_delta = b.output() - b_grad;
        self.graph.mut_node_output(3, b_delta); 
    }

}


impl ModelSerialize for SGD {

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = SGDSerialize {
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


        let obj = SGDSerialize {
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
        let obj: SGDSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(SGD {
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

        let obj: SGDSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(SGD {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            weight_dim: obj.weight_dim,
            bias_dim: obj.bias_dim,
            learning_rate: obj.learning_rate
        })
    }

}
