use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*;
use crate::loss::mse::*;
use crate::regression::linear::base::Linear;


#[derive(Debug, Clone, PartialEq)]
pub struct SimpleLinear {
    inputs: NDArray<f64>,
    outputs: NDArray<f64>,
    slope: f64, 
    bias: f64,
    learning_rate: f64,
    loss_function: fn(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<f64, String>
}


impl SimpleLinear {

    pub fn new(x: NDArray<f64>, y: NDArray<f64>, learning_rate: f64) -> Result<SimpleLinear, String> {
        Ok(Self {
            inputs: x.clone(),
            outputs: y.clone(),
            slope: 0.00,
            bias: 0.00,
            learning_rate: learning_rate,
            loss_function: mse
        })
    }

}

impl Linear for SimpleLinear {

    fn set_loss(&mut self, loss_func: fn(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<f64, String>) {
        self.loss_function = loss_func;
    }


    fn predict(&mut self, inputs: NDArray<f64>) -> Result<NDArray<f64>, String> {

        if inputs.shape() != self.inputs.shape() {
            return Err("Provided shape values don't match model params".to_string());
        }

        self.inputs = inputs; 
        let y_pred = self.forward().unwrap();
        Ok(y_pred)
    }


    fn forward(&self) -> Result<NDArray<f64>, String> {
        let slope_mult = self.inputs.scalar_mult(self.slope).unwrap(); 
        let bias_add = slope_mult.scalar_add(self.bias).unwrap(); 
        Ok(bias_add)
    }

    fn weight_update(&mut self, y_pred: NDArray<f64>)  {
        let grad = -2.0/y_pred.size() as f64;
        let db_dm = self.inputs.mult(y_pred).unwrap(); 
        let dm_sum: f64 = db_dm.scalar_mult(grad).unwrap().values().iter().sum();
        self.slope = self.slope - self.learning_rate * dm_sum;
    }


    fn bias_update(&mut self, y_pred: NDArray<f64>) {
        let grad = -2.0/y_pred.size() as f64;
        let db: f64 = y_pred.scalar_mult(grad).unwrap().values().iter().sum();
        self.bias = self.bias - self.learning_rate * db;
    }

    fn train(&mut self, epochs: usize, log_output: bool) {

        for epoch in 0..epochs {

            let result = self.forward().unwrap(); 
            let output_error = self.outputs.subtract(result.clone()).unwrap();
            let loss = mse(result.clone(), self.outputs.clone()).unwrap(); 
            
            self.weight_update(output_error.clone());
            self.bias_update(output_error);

            if log_output {
                println!("Epoch [{:?}/{:?}]: {:?}", epoch, epochs, loss);
            }


        }
    }

}