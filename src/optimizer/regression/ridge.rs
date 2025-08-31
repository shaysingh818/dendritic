//! Ridge regression model implementation

use std::fs;
use std::fs::File; 
use std::io::{Write, BufWriter, BufReader}; 

use uuid::Uuid;
use chrono::{Datelike, Utc};  
use ndarray::{Array2};
use serde::{Serialize, Deserialize}; 

use crate::autodiff::graph::{ComputationGraph, GraphSerialize};

use crate::optimizer::model::*; 
use crate::optimizer::regression::sgd::*; 


/// Ridge regression
pub struct Ridge {
    
    /// Instance of linear regression structure
    pub sgd: SGD,

    /// lambda parameter to regularize weights
    pub lambda: f64
}

/// Serialization structure for ridge regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeSerialize {

    /// Serializable instance of regression structure
    sgd: SGDSerialize,

    /// lambda parmeter to regualrize weights
    lambda: f64
}


impl Ridge {


    /// Create instance of ridge regression model.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features for training.
    /// * `y` - Target labels for training.
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `lambda` - The regularization strength.
    ///
    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64, 
        lambda: f64) -> Result<Self, String> {

        Ok(Self {
            sgd: SGD::new(x, y, learning_rate).unwrap(),
            lambda: lambda
        })
    }

}


impl Model for Ridge {
    
    fn input(&self) -> Array2<f64> {
        self.sgd.input()
    }

    fn output(&self) -> Array2<f64> {
        self.sgd.output()
    }

    fn set_input(&mut self, x: &Array2<f64>) {
        self.sgd.set_input(x);
    }

    fn set_output(&mut self, y: &Array2<f64>) {
        self.sgd.set_output(y);
    }

    fn graph(&self) -> &ComputationGraph<Array2<f64>> {
        &self.sgd.graph
    }

    fn forward(&mut self) {
        self.sgd.forward();
    }

    fn backward(&mut self) {
        self.sgd.backward();
    }

    fn predicted(&self) -> Array2<f64> {
        self.sgd.predicted()
    }

    fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.sgd.predict(x)
    }

    fn loss(&mut self) -> f64 {
        let loss_node = self.sgd.graph.curr_node();
        let loss = loss_node.output();
        let weights = self.sgd.graph.node(1).output();
        let l2 = weights.mapv(|x| (x as f64).powf(2.0)).sum();
        let loss_val = loss.clone() + (self.lambda * l2);
        loss_val.as_slice().unwrap()[0]
    }

    fn set_loss(&mut self, op: Box<dyn Operation<Array2<f64>>>) {
        self.sgd.set_loss(op);
    }
 
    fn update_parameters(&mut self) {

        let lr = self.sgd.learning_rate;
        let w = self.sgd.graph.node(1);
        let w_grad = w.grad() + self.lambda * 2.0 * w.output();
        let w_delta = w.output() - lr * w_grad;
        self.sgd.graph.mut_node_output(1, w_delta); 

        let b = self.sgd.graph.node(3);
        let b_grad = b.grad() * lr;
        let b_delta = b.output() - b_grad;
        self.sgd.graph.mut_node_output(3, b_delta); 
    }

    fn update_parameter(&mut self, idx: usize, val: Array2<f64>) {
        self.sgd.update_parameter(idx, val);
    }

}


impl ModelSerialize for Ridge {

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = RidgeSerialize {
            sgd: SGDSerialize {
                graph_path: format!("{filepath}/regression_exp"),
                weight_dim: self.sgd.weight_dim,
                bias_dim: self.sgd.bias_dim,
                learning_rate: self.sgd.learning_rate
            },
            lambda: self.lambda
        };

        let _ = self.sgd.graph.save(&obj.sgd.graph_path); 
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

        let obj = RidgeSerialize {
            sgd: SGDSerialize {
                graph_path: format!("{namespace}/regression_exp"),
                weight_dim: self.sgd.weight_dim,
                bias_dim: self.sgd.bias_dim,
                learning_rate: self.sgd.learning_rate
            },
            lambda: self.lambda
        };

        let _ = self.sgd.graph.save(&obj.sgd.graph_path); 
        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file); 
        let json_string = serde_json::to_string_pretty(&obj)?;
        writer.write_all(json_string.as_bytes())?; 
        Ok(())
    }
 
    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {

        let parameter_path = format!("{filepath}/parameters.json");
        let obj: RidgeSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        let sgd = SGD {
            graph: ComputationGraph::load(&obj.sgd.graph_path).unwrap(),
            weight_dim: obj.sgd.weight_dim,
            bias_dim: obj.sgd.bias_dim,
            learning_rate: obj.sgd.learning_rate
        };

        Ok(Ridge {
            sgd: sgd,
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

        let obj: RidgeSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        let sgd = SGD {
            graph: ComputationGraph::load(&obj.sgd.graph_path).unwrap(),
            weight_dim: obj.sgd.weight_dim,
            bias_dim: obj.sgd.bias_dim,
            learning_rate: obj.sgd.learning_rate
        };

        Ok(Ridge {
            sgd: sgd,
            lambda: obj.lambda
        })
    }

}
