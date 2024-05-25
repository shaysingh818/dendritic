use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*;
use crate::loss::mse::*;


#[derive(Debug, Clone, PartialEq)]
pub struct Ridge {
    features: NDArray<f64>,
    outputs: NDArray<f64>,
    predicted_outputs: NDArray<f64>,
    weights: NDArray<f64>, 
    bias: f64,
    learning_rate: f64,
    loss_function: fn(y_true: &NDArray<f64>, y_pred: &NDArray<f64>) -> Result<f64, String>,
    model_loss: f64,
    lambda: f64
}


impl Ridge {


    pub fn new(features: NDArray<f64>, y: NDArray<f64>, learning_rate: f64) -> Result<Ridge, String> {

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
            loss_function: mse,
            model_loss: 0.00,
            lambda: 1.0
        })
    }


    pub fn forward(&self) -> Result<NDArray<f64>, String> {
        let result = self.features.dot(self.weights.clone()).unwrap();
        let bias_add = result.scalar_add(self.bias).unwrap();
        Ok(bias_add)
    }

    pub fn weight_update(&mut self, _y_pred: NDArray<f64>) {
        // self.weights = self.weights.subtract(d_w).unwrap(); 
    }

    pub fn bias_update(&mut self, _y_pred: NDArray<f64>)  {
        // self.bias = self.bias - db; 
    }

    pub fn shrinkage_penalty(&self) -> f64 {
        let mut sum = 0.0; 
        for item in self.weights.values() {
            sum += item.powf(2.0);
        }
        sum
    }


    pub fn train(&mut self, epochs: usize, log_output: bool) {
        self.forward().unwrap();
        for epoch in 0..epochs {
            let y_pred = self.forward().unwrap(); 
            let mse = (self.loss_function)(&y_pred, &self.outputs).unwrap();
            let penalty =  self.shrinkage_penalty();
            let loss = mse + penalty; 

            self.weight_update(y_pred.clone());
            self.bias_update(y_pred); 

            if log_output {
                println!("Epoch [{:?}/{:?}]: {:?}", epoch, epochs, loss);
            }
        }
    }


}