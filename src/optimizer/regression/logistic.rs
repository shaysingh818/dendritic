//! Logistic regression model implementation

use std::fs;
use std::fs::File; 
use std::io::{Write, BufWriter, BufReader}; 

use uuid::Uuid;
use chrono::{Datelike, Utc};  
use ndarray::{stack,  Array2, Axis};
use serde::{Serialize, Deserialize}; 

use crate::autodiff::operations::arithmetic::*; 
use crate::autodiff::operations::loss::*;
use crate::autodiff::operations::activation::*;
use crate::autodiff::operations::base::Operation; 
use crate::autodiff::graph::{ComputationGraph, GraphConstruction, GraphSerialize};

use crate::optimizer::model::*;


/// Logistic regression classifier
pub struct Logistic {

    /// Underlying computation graph with operations for optimizer
    pub graph: ComputationGraph<Array2<f64>>,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Learning rate to control how fast to decrease
    pub learning_rate: f64,

    /// Flag for multi or binary classification
    pub multi_class: bool
}


/// Serialization structure for logistic regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticSerialize {

    /// Underlying computation graph with operations for optimizer
    pub graph_path: String,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Learning rate to control how fast to decrease
    pub learning_rate: f64,

    /// Serialized flag for multi or binary classification
    pub multi_class: bool
}


impl Logistic {


    /// Create instance of logistic regression model.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features for training.
    /// * `y` - Target labels for training.
    /// * `multi_class` - Flag for multi-class classification.
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        multi_class: bool,
        learning_rate: f64) -> Result<Self, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }

        let mut weight_dim: (usize, usize) = (x.shape()[1], 1);
        let mut bias_dim: (usize, usize) = (1, y.shape()[1]);

        if multi_class {
            weight_dim = (x.shape()[1], y.shape()[1]);
            bias_dim = (1, y.shape()[1]);
        }
            
        let mut log = Logistic {
            graph: ComputationGraph::new(),
            weight_dim: weight_dim,
            bias_dim: bias_dim,
            learning_rate: learning_rate,
            multi_class: multi_class
        };

        log.graph.mul(vec![x.clone(), Array2::zeros(log.weight_dim)]);
        log.graph.add(vec![Array2::zeros(log.bias_dim)]);

        if log.multi_class {
            log.graph.cce(y.clone()); 
        } else {
            log.graph.sigmoid();
            log.graph.bce(y.clone()); 
        }

        log.graph.add_parameter(1);
        log.graph.add_parameter(3);
        Ok(log)
    }

}


impl Model for Logistic {
    
    fn input(&self) -> Array2<f64> {
        self.graph.node(0).output()
    }

    fn output(&self) -> Array2<f64> {
        if self.multi_class {
            self.graph.node(5).output()
        } else {
            self.graph.node(6).output()
        }
    }

    fn set_input(&mut self, x: &Array2<f64>) {
        self.graph.mut_node_output(0, x.to_owned());
    }

    fn set_output(&mut self, y: &Array2<f64>) {
        if self.multi_class {
            self.graph.mut_node_output(4, y.to_owned());
            self.graph.mut_node_output(5, y.to_owned());
        } else {
            self.graph.mut_node_output(6, y.to_owned());
            self.graph.mut_node_output(7, y.to_owned());
        }
    }

    fn graph(&self) -> &ComputationGraph<Array2<f64>> {
        &self.graph
    }

    fn forward(&mut self) {
        self.graph.forward();
    }

    fn backward(&mut self) {
        self.graph.backward();
    }

    fn predicted(&self) -> Array2<f64> {

        if self.multi_class {
            let mut row_idx = 0;
            let mut predictions = Array2::zeros((self.output().nrows(), 2));
            let output = self.graph.node(4).output(); // bias node

            let samples: Vec<_> = output
                .axis_iter(Axis(0))
                .map(|row| {
                    let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp = row.mapv(|x| (x - max).exp());
                    let sum = exp.sum();
                    exp.mapv(|x| x / sum)
                })
                .collect();

            let views: Vec<_> = samples.iter().map(|r| r.view()).collect();
            let softmax = stack(Axis(0), &views).unwrap();

            for row in softmax.axis_iter(Axis(0)) {
                let (predicted_idx, &prob) = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                predictions[[row_idx, 0]] = predicted_idx as f64;
                predictions[[row_idx, 1]] = prob;
                row_idx += 1; 
            }
            predictions
        } else {
            self.graph.node(5).output()
        }
    }

    fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.set_output(&Array2::zeros((x.nrows(), self.output().dim().1)));
        self.set_input(x);
        self.graph.forward();
        self.predicted()
    }

    fn loss(&mut self) -> f64 {
        let loss_node = self.graph.curr_node();
        let loss = loss_node.output();
        loss.as_slice().unwrap()[0]
    }


    fn set_loss(&mut self, op: Box<dyn Operation<Array2<f64>>>) {
        let idx = self.graph.nodes().len() - 1;
        self.graph.nodes[idx].set_operation(op);
    }
    
    fn update_parameters(&mut self) {

        let w = self.graph.node(1);
        let w_grad = w.grad() * self.learning_rate;
        let w_delta = w.output() - w_grad;
        self.graph.mut_node_output(1, w_delta); 

        let b = self.graph.node(3);
        let b_grad = b.grad() * self.learning_rate;
        let b_delta = b.output() - b_grad.clone();
        self.graph.mut_node_output(3, b_delta); 

    }

    fn update_parameter(&mut self, idx: usize, val: Array2<f64>) {
        self.graph.mut_node_output(idx, val);
    }

}


impl ModelSerialize for Logistic {

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = LogisticSerialize {
            graph_path: format!("{filepath}/regression_exp"),
            weight_dim: self.weight_dim,
            bias_dim: self.bias_dim,
            learning_rate: self.learning_rate,
            multi_class: self.multi_class
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
            learning_rate: self.learning_rate,
            multi_class: self.multi_class
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
            learning_rate: obj.learning_rate,
            multi_class: obj.multi_class
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
            learning_rate: obj.learning_rate,
            multi_class: obj.multi_class
        })
    }

}
