use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use metrics::loss::*;
use autodiff::node::{Node, Value};
use autodiff::ops::*; 
use std::fs;

pub struct Linear {
    pub features: Value<NDArray<f64>>,
    pub outputs: Value<NDArray<f64>>,
    pub weights: Value<NDArray<f64>>, 
    pub bias: Value<NDArray<f64>>,
    learning_rate: f64,
    loss_function: fn(
        y_true: &NDArray<f64>, 
        y_pred: &NDArray<f64>) -> Result<f64, String>
}


impl Linear {

    pub fn new(
        features: NDArray<f64>, 
        y: NDArray<f64>, 
        learning_rate: f64) -> Result<Linear, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err("Learning rate must be between 1 and 0".to_string());
        }

        let weights = NDArray::new(vec![features.shape().dim(1), 1]).unwrap();
        let bias = NDArray::new(vec![1, 1]).unwrap();
        let inputs = Value::new(&features); 
        let outputs = Value::new(&y);

        Ok(Self {
            features: inputs.clone(),
            outputs: outputs.clone(),
            weights: Value::new(&weights),
            bias: Value::new(&bias),
            learning_rate: learning_rate,
            loss_function: mse,
        })
    }

    pub fn predict(&mut self, inputs: NDArray<f64>) -> NDArray<f64> {

        self.features = Value::new(&inputs); 

        let mut linear = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );

        linear.forward(); 
        linear.value()
        
    }


    pub fn save(&self, filepath: &str) -> std::io::Result<()> {

        let weights_file = format!("{}/weights", filepath);
        let bias_path = format!("{}/bias", filepath); 
        fs::create_dir_all(filepath)?;

        self.weights.val().save(&weights_file).unwrap();
        self.bias.val().save(&bias_path).unwrap();

        Ok(())
    }


    pub fn load(
        filepath: &str, 
        features: NDArray<f64>, 
        y: NDArray<f64>, 
        learning_rate: f64) -> std::io::Result<Linear> {

        let weights_file = format!("{}/weights", filepath);
        let bias_path = format!("{}/bias", filepath); 

        let inputs = Value::new(&features); 
        let outputs = Value::new(&y);
        let load_weights = NDArray::load(&weights_file).unwrap();
        let load_bias = NDArray::load(&bias_path).unwrap();

        Ok(Linear {
            features: inputs.clone(),
            outputs: outputs.clone(),
            weights: Value::new(&load_weights),
            bias: Value::new(&load_bias),
            learning_rate: learning_rate,
            loss_function: mse,
        })

    }


    pub fn train(&mut self, epochs: usize, log_output: bool) {
        
        /* create node graph */     
        let mut linear = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );

        for epoch in 0..epochs {

            linear.forward();

            let y_pred = linear.value();
            let loss = (self.loss_function)(&self.outputs.val(), &y_pred);
            let error = y_pred.subtract(self.outputs.val()).unwrap();

            linear.backward(error);

            /* update weights */
            let learning_rate_factor = self.learning_rate/y_pred.size() 
                as f64;

            let w_grad = self.features
                .grad()
                .scalar_mult(learning_rate_factor)
                .unwrap();

            let dw = self.weights.val().subtract(w_grad).unwrap();
            self.weights.set_val(&dw); 

            /* update biases */
            let b_collapse = self.bias
                .grad()
                .sum_axis(1)
                .unwrap();

            let db = b_collapse.scalar_mult(learning_rate_factor).unwrap();
            self.bias.set_val(&db);

            if log_output {
                println!("Epoch [{:?}/{:?}]: {:?}", epoch, epochs, loss);
            }

        }

    }


    pub fn sgd(&mut self, epochs: usize, log_output: bool, batch_size: usize) {
        
        let mut loss: f64 = 0.0;
        let x_train_binding = self.features.val();
        let y_train_binding = self.outputs.val();
        let x_train = x_train_binding.batch(batch_size).unwrap();
        let y_train = y_train_binding.batch(batch_size).unwrap();

        let mut linear = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );

        for epoch in 0..epochs {

            let mut batch_index = 0;
            for batch in &x_train {

                self.features.set_val(&batch);
                self.outputs.set_val(&y_train[batch_index]);

                linear.forward();

                let y_pred = linear.value();
                loss = (self.loss_function)(&self.outputs.val(), &y_pred).unwrap();
                let error = y_pred.subtract(self.outputs.val()).unwrap();

                linear.backward(error);

                /* update weights */
                let learning_rate_factor = self.learning_rate/y_pred.size() 
                    as f64;

                let w_grad = self.features
                    .grad()
                    .scalar_mult(learning_rate_factor)
                    .unwrap();

                let dw = self.weights.val().subtract(w_grad).unwrap();
                self.weights.set_val(&dw); 

                /* update biases */
                let b_collapse = self.bias
                    .grad()
                    .sum_axis(1)
                    .unwrap();

                let db = b_collapse.scalar_mult(learning_rate_factor).unwrap();
                self.bias.set_val(&db);

                batch_index += 1; 
            }

            if log_output {
                println!("Epoch [{:?}/{:?}]: {:?}", epoch, epochs, loss);
            }
            
        }

    }





}
