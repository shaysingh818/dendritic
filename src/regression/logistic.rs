use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*;
use crate::loss::sigmoid::*;
use crate::loss::mse::*;

use std::fs;
use std::fs::File;
use std::io::{Write, Read};
use serde_json::{to_string, from_str, Value};  

#[derive(Debug, Clone, PartialEq)]
pub struct Logistic {
    features: NDArray<f64>,
    outputs: NDArray<f64>,
    predicted_outputs: NDArray<f64>,
    weights: NDArray<f64>, 
    bias: f64,
    learning_rate: f64,
    activation_function: fn(values: f64) -> f64,
    loss_function: fn(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<f64, String>,
    model_loss: f64
}


impl Logistic {

    pub fn weights(&self) -> &NDArray<f64> {
        &self.weights
    }

    pub fn outputs(&self) -> &NDArray<f64> {
        &self.outputs
    }

    pub fn predicted(&self) -> &NDArray<f64> {
        &self.predicted_outputs
    }

    pub fn loss(&self) -> f64 {
        self.model_loss
    }

    pub fn new(features: NDArray<f64>, y: NDArray<f64>, learning_rate: f64) -> Result<Logistic, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err("Learning rate must be between 1 and 0".to_string());
        }

        Ok(Self {
            features: features.clone(),
            outputs: y.clone(),
            predicted_outputs: NDArray::new(y.shape().to_vec()).unwrap(),
            weights: NDArray::new(vec![features.shape()[1], 1]).unwrap(),
            bias: 0.00,
            learning_rate: learning_rate,
            activation_function: sigmoid,
            loss_function: mse,
            model_loss: 0.00
        })
    }

    pub fn save(&self, filepath: &str) -> std::io::Result<()> {

        let bias_str = to_string(&self.bias)?;
        let weights_file = format!("{}/weights", filepath);
        let bias_path = format!("{}/bias.json", filepath); 
        fs::create_dir_all(filepath)?;

        self.weights.save(&weights_file).unwrap();

        let mut bias_file = File::create(bias_path)?;
        bias_file.write_all(bias_str.as_bytes())?;

        Ok(())

    }

    pub fn load(
        filepath: &str, 
        features: NDArray<f64>, 
        y: NDArray<f64>, 
        learning_rate: f64) -> std::io::Result<Logistic> {

        let weights_file = format!("{}/weights", filepath);
        let bias_path = format!("{}/bias.json", filepath); 
        let mut file = File::open(bias_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let bias_value: f64 = from_str(&contents)?;

        Ok(Logistic {
            features: features.clone(),
            outputs: y.clone(),
            predicted_outputs: NDArray::new(y.shape().to_vec()).unwrap(),
            weights: NDArray::load(&weights_file).unwrap(),
            bias: bias_value,
            learning_rate: learning_rate,
            activation_function: sigmoid,
            loss_function: mse,
            model_loss: 0.00
        })
    }


    pub fn forward(&self) -> Result<NDArray<f64>, String> {
        let result = self.features.dot(self.weights.clone()).unwrap();
        let bias_add = result.scalar_add(self.bias).unwrap();
        let add_loss = bias_add.apply(self.activation_function).unwrap();
        Ok(add_loss)
    }

    pub fn predict(&mut self, input_features: NDArray<f64>, outputs: NDArray<f64>) -> NDArray<f64> {
        self.features = input_features;
        self.outputs = outputs;
        self.predicted_outputs = self.forward().unwrap();
        self.predicted_outputs.clone()
    }


    fn weight_update(&mut self, y_pred: NDArray<f64>) {
        let x_t = self.features.clone().transpose().unwrap();
        let error = y_pred.subtract(self.outputs.clone()).unwrap();
        let grad = x_t.dot(error).unwrap();
        let d_w = grad.scalar_mult(self.learning_rate / y_pred.size() as f64).unwrap(); 
        self.weights = self.weights.subtract(d_w).unwrap(); 
    }

    fn bias_update(&mut self, y_pred: NDArray<f64>)  {
        let error = y_pred.subtract(self.outputs.clone()).unwrap();
        let grad = self.learning_rate/y_pred.size() as f64;
        let db: f64 = error.scalar_mult(grad).unwrap().values().iter().sum();
        self.bias = self.bias - db; 
    }


    pub fn train(&mut self, epochs: usize, log_output: bool, batch_size: usize) {

        let mut loss: f64 = 0.0;

        if batch_size > 0 {

            let mut input_train: Vec<NDArray<f64>> = self.features.batch(batch_size).unwrap();
            let mut output_train: Vec<NDArray<f64>> = self.outputs.batch(batch_size).unwrap();

            for epoch in 0..epochs {

                let mut batch_index = 0; 
                for batch in &input_train {

                    self.features = batch.clone(); 
                    self.outputs = output_train[batch_index].clone();

                    let mut y_pred = self.forward().unwrap();
                    loss = (self.loss_function)(y_pred.clone(), self.outputs.clone()).unwrap(); 
                    self.weight_update(y_pred.clone());
                    self.bias_update(y_pred.clone()); 
                    batch_index += 1; 
                }

                if log_output {
                    println!("Epoch [{:?}/{:?}]: {:?}", epoch, epochs, loss);
                }
            }

            self.model_loss = loss;

        } else {

            let mut y_pred = self.forward().unwrap();
            for epoch in 0..epochs {
                y_pred = self.forward().unwrap(); 
                let loss = (self.loss_function)(y_pred.clone(), self.outputs.clone()).unwrap(); 
                self.weight_update(y_pred.clone());
                self.bias_update(y_pred.clone()); 

                if log_output {
                    println!("Epoch [{:?}/{:?}]: {:?}", epoch, epochs, loss);
                }
            }

            self.model_loss = loss; 

        }

    }


}