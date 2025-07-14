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


pub struct Regression {

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
pub struct RegressionSerialize {

    /// Underlying computation graph with operations for optimizer
    pub graph_path: String,

    /// Coefficients associated with each feature
    pub weight_dim: (usize, usize),

    /// Bias to add after weights multiplication
    pub bias_dim: (usize, usize),

    /// Learning rate to control how fast to decrease
    pub learning_rate: f64

}


impl Regression {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64) -> Result<Self, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err(
                "Learning rate must be between 0 and 1".to_string()
            );
        }

        let mut regression = Regression {
            graph: ComputationGraph::new(),
            weight_dim: (x.shape()[1], 1),
            bias_dim: (1, 1),
            learning_rate: learning_rate
        };

        regression.graph.mul(vec![x.clone(), Array2::zeros(regression.weight_dim)]); 
        regression.graph.add(vec![Array2::zeros(regression.bias_dim)]);
        regression.graph.mse(y.clone());

        regression.graph.add_parameter(1);
        regression.graph.add_parameter(3);
        Ok(regression)
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn input(&self) -> Array2<f64> {
        self.graph.node(0).output()
    }

    pub fn output(&self) -> Array2<f64> {
        self.graph.node(5).output()
    }

    pub fn set_input(&mut self, x: &Array2<f64>) {
        self.graph.mut_node_output(0, x.to_owned());
    }

    pub fn set_output(&mut self, y: &Array2<f64>) {
        self.graph.mut_node_output(4, y.to_owned());
        self.graph.mut_node_output(5, y.to_owned());
    }

    pub fn fit(&mut self, epochs: usize) {

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
    }

    pub fn fit_batch(
        &mut self, 
        x_train: Array2<f64>, 
        y_train: Array2<f64>,
        batch_size: usize,
        num_batches: usize,
        rows: usize) {

        let bar = ProgressBar::new(1000);
        bar.set_style(ProgressStyle::default_bar()
            .template("{bar:50} {pos}/{len}")
            .unwrap());

        for _ in 0..1000 {

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

                self.set_input(&x.to_owned());
                self.set_output(&y.to_owned());

                self.graph.forward();
                self.graph.backward(); 
                self.parameter_update();
            }
            bar.inc(1); 
        }

        bar.finish(); 


    }

}

/// Trait for shared methods across different types of regression models
/// Base regression structure is used across all these methods
pub trait RegressionOptimizer {

    /// Method for updating parameters across different optimizers
    fn parameter_update(&mut self);

    /// Method for measuring loss across different types of regression models
    fn measure_loss(&mut self) -> f64;

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


impl RegressionOptimizer for Regression {

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

        let obj = RegressionSerialize {
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


        let obj = RegressionSerialize {
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
        let obj: RegressionSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(Regression {
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

        let obj: RegressionSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        Ok(Regression {
            graph: ComputationGraph::load(&obj.graph_path).unwrap(),
            weight_dim: obj.weight_dim,
            bias_dim: obj.bias_dim,
            learning_rate: obj.learning_rate
        })
    }

}


pub struct Ridge {
    
    /// Instance of linear regression structure
    pub regression: Regression,

    /// lambda parameter to regularize weights
    pub lambda: f64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeSerialize {

    /// Serializable instance of regression structure
    regression: RegressionSerialize,

    /// lambda parmeter to regualrize weights
    lambda: f64
}


impl Ridge {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64, 
        lambda: f64) -> Result<Self, String> {

        Ok(Self {
            regression: Regression::new(x, y, learning_rate).unwrap(),
            lambda: lambda
        })
    }

}


impl RegressionOptimizer for Ridge {

    fn parameter_update(&mut self) {

        let lr = self.regression.learning_rate;
        let w = self.regression.graph.node(1);
        let w_grad = w.grad() + self.lambda * 2.0 * w.output();
        let w_delta = w.output() - lr * w_grad;
        self.regression.graph.mut_node_output(1, w_delta); 

        let b = self.regression.graph.node(3);
        let b_grad = (b.grad() * lr).sum_axis(Axis(0));
        let b_delta = b.output() - b_grad;
        self.regression.graph.mut_node_output(3, b_delta); 
    }

    fn measure_loss(&mut self) -> f64 {

        let loss_node = self.regression.graph.curr_node();
        let loss = loss_node.output();
        let weights = self.regression.graph.node(1).output();
        let l2 = weights.mapv(|x| (x as f64).powf(2.0)).sum();
        let loss_val = loss.clone() + (self.lambda * l2);
        loss_val.as_slice().unwrap()[0]
    }

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = RidgeSerialize {
            regression: RegressionSerialize {
                graph_path: format!("{filepath}/regression_exp"),
                weight_dim: self.regression.weight_dim,
                bias_dim: self.regression.bias_dim,
                learning_rate: self.regression.learning_rate
            },
            lambda: self.lambda
        };

        let _ = self.regression.graph.save(&obj.regression.graph_path); 
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
            regression: RegressionSerialize {
                graph_path: format!("{namespace}/regression_exp"),
                weight_dim: self.regression.weight_dim,
                bias_dim: self.regression.bias_dim,
                learning_rate: self.regression.learning_rate
            },
            lambda: self.lambda
        };

        let _ = self.regression.graph.save(&obj.regression.graph_path); 
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

        let reg = Regression {
            graph: ComputationGraph::load(&obj.regression.graph_path).unwrap(),
            weight_dim: obj.regression.weight_dim,
            bias_dim: obj.regression.bias_dim,
            learning_rate: obj.regression.learning_rate
        };

        Ok(Ridge {
            regression: reg,
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

        let reg = Regression {
            graph: ComputationGraph::load(&obj.regression.graph_path).unwrap(),
            weight_dim: obj.regression.weight_dim,
            bias_dim: obj.regression.bias_dim,
            learning_rate: obj.regression.learning_rate
        };

        Ok(Ridge {
            regression: reg,
            lambda: obj.lambda
        })
    }

}



pub struct Lasso {
    
    /// Instance of linear regression structure
    pub regression: Regression,

    /// lambda parameter to regularize weights
    pub lambda: f64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LassoSerialize {

    /// Serializable instance of regression structure
    regression: RegressionSerialize,

    /// lambda parmeter to regualrize weights
    lambda: f64
}


impl Lasso {

    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64, 
        lambda: f64) -> Result<Self, String> {

        Ok(Self {
            regression: Regression::new(x, y, learning_rate).unwrap(),
            lambda: lambda
        })
    }

}


impl RegressionOptimizer for Lasso {

    fn parameter_update(&mut self) {

        let lr = self.regression.learning_rate;
        let w = self.regression.graph.node(1);
        let sig_w = self.regression.graph.node(1).output().mapv(|x| x.signum());
        let w_grad = w.clone().grad() + (self.lambda * sig_w);
        let w_new = w.output() - (w_grad * lr);
        self.regression.graph.mut_node_output(1, w_new); 

        let b = self.regression.graph.node(3);
        let b_grad = (b.grad() * lr).sum_axis(Axis(0));
        let b_delta = b.output() - b_grad;
        self.regression.graph.mut_node_output(3, b_delta);  

    }

    fn measure_loss(&mut self) -> f64 {

        let loss_node = self.regression.graph.curr_node();
        let loss = loss_node.output();
        let weights = self.regression.graph.node(1).output();
        let l1 = weights.mapv(|x| x.abs()).sum();
        let loss_val = loss.clone() + (self.lambda * l1);
        loss_val.as_slice().unwrap()[0]
    }

    fn save(&self, filepath: &str) -> std::io::Result<()> {

        fs::create_dir_all(filepath)?;
        let file_path = format!("{filepath}/parameters.json");

        let obj = LassoSerialize {
            regression: RegressionSerialize {
                graph_path: format!("{filepath}/regression_exp"),
                weight_dim: self.regression.weight_dim,
                bias_dim: self.regression.bias_dim,
                learning_rate: self.regression.learning_rate
            },
            lambda: self.lambda
        };

        let _ = self.regression.graph.save(&obj.regression.graph_path); 
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

        let obj = LassoSerialize {
            regression: RegressionSerialize {
                graph_path: format!("{namespace}/regression_exp"),
                weight_dim: self.regression.weight_dim,
                bias_dim: self.regression.bias_dim,
                learning_rate: self.regression.learning_rate
            },
            lambda: self.lambda
        };

        let _ = self.regression.graph.save(&obj.regression.graph_path); 
        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file); 
        let json_string = serde_json::to_string_pretty(&obj)?;
        writer.write_all(json_string.as_bytes())?; 
        Ok(())
    }
 
    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {

        let parameter_path = format!("{filepath}/parameters.json");
        let obj: LassoSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        let reg = Regression {
            graph: ComputationGraph::load(&obj.regression.graph_path).unwrap(),
            weight_dim: obj.regression.weight_dim,
            bias_dim: obj.regression.bias_dim,
            learning_rate: obj.regression.learning_rate
        };

        Ok(Lasso {
            regression: reg,
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

        let obj: LassoSerialize = {
            let file = File::open(&parameter_path)?; 
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };

        let reg = Regression {
            graph: ComputationGraph::load(&obj.regression.graph_path).unwrap(),
            weight_dim: obj.regression.weight_dim,
            bias_dim: obj.regression.bias_dim,
            learning_rate: obj.regression.learning_rate
        };

        Ok(Lasso {
            regression: reg,
            lambda: obj.lambda
        })
    }

}
