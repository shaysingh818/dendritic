use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use metrics::activations::*;
use metrics::loss::*;
use metrics::utils::*; 
use autodiff::node::{Node, Value};
use autodiff::ops::*; 
use std::fs;

pub struct Logistic {
    pub features: Value<NDArray<f64>>,
    pub outputs: Value<NDArray<f64>>,
    pub weights: Value<NDArray<f64>>, 
    pub bias: Value<NDArray<f64>>,
    learning_rate: f64,
    activation_function: fn(values: NDArray<f64>) -> NDArray<f64>,
    loss_function: fn(
        y_true: &NDArray<f64>, 
        y_pred: &NDArray<f64>) -> Result<f64, String>
}


impl Logistic {

    pub fn new(
        features: &NDArray<f64>, 
        y: &NDArray<f64>,
        activation_function: fn(values: NDArray<f64>) -> NDArray<f64>,
        learning_rate: f64) -> Result<Logistic, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err("Learning rate must be between 1 and 0".to_string());
        }

    
        let weights_shape = vec![features.shape().dim(1), y.shape().dim(1)];
        let weights = NDArray::new(vec![features.shape().dim(1), 1]).unwrap();
        let bias = NDArray::new(vec![1, 1]).unwrap();
        let inputs = Value::new(features); 
        let outputs = Value::new(y);

        Ok(Self {
            features: inputs,
            outputs: outputs,
            weights: Value::new(&weights),
            bias: Value::new(&bias),
            learning_rate: learning_rate,
            activation_function: activation_function, 
            loss_function: binary_cross_entropy // default loss function
        })
    }

    pub fn predict(&mut self, inputs: NDArray<f64>) -> NDArray<f64> {

        self.features = Value::new(&inputs); 

        let mut linear = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );

        linear.forward();
        (self.activation_function)(linear.value())
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
        features: &NDArray<f64>, 
        y: &NDArray<f64>,
        activation_function: fn(values: NDArray<f64>) -> NDArray<f64>,
        learning_rate: f64) -> std::io::Result<Logistic> {

        let weights_file = format!("{}/weights", filepath);
        let bias_path = format!("{}/bias", filepath); 

        let inputs = Value::new(features); 
        let outputs = Value::new(y);
        let load_weights = NDArray::load(&weights_file).unwrap();
        let load_bias = NDArray::load(&bias_path).unwrap();

        Ok(Logistic {
            features: inputs,
            outputs: outputs,
            weights: Value::new(&load_weights),
            bias: Value::new(&load_bias),
            learning_rate: learning_rate,
            activation_function: activation_function,
            loss_function: mse,
        })

    }


    pub fn train(&mut self, epochs: usize, log_output: bool) {

        /* create node graph */     
        let mut logistic = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );


        for epoch in 0..epochs {

            logistic.forward();

            let y_pred = (self.activation_function)(logistic.value());
            let loss = (self.loss_function)(&y_pred, &self.outputs.val()).unwrap();
            let error = y_pred.subtract(self.outputs.val()).unwrap();
            logistic.backward(error);

            /* update weights */
            let learning_rate_factor = (1.0/y_pred.size() as f64) * self.learning_rate;

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

        let mut logistic = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        ); 

        for epoch in 0..epochs {

            let mut batch_index = 0;
            for batch in &x_train {

                self.features.set_val(&batch);
                self.outputs.set_val(&y_train[batch_index]);

                logistic.forward();

                let y_pred = (self.activation_function)(logistic.value());
                loss = (self.loss_function)(&y_pred, &self.outputs.val()).unwrap();
                let error = y_pred.subtract(self.outputs.val()).unwrap();

                logistic.backward(error);

                /* update weights */
                let learning_rate_factor = self.learning_rate/y_pred.size() as f64;

                let w_grad = self.features
                    .grad()
                    .scalar_mult(learning_rate_factor)
                    .unwrap();

                let dw = self.weights.val().subtract(w_grad).unwrap();
                self.weights.set_val(&dw); 

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


pub struct MultiClassLogistic {
    pub features: Value<NDArray<f64>>,
    pub outputs: Value<NDArray<f64>>,
    pub weights: Value<NDArray<f64>>, 
    pub bias: Value<NDArray<f64>>,
    learning_rate: f64,
    activation_function: fn(values: NDArray<f64>) -> NDArray<f64>,
    loss_function: fn(
        y_true: &NDArray<f64>, 
        y_pred: &NDArray<f64>) -> Result<f64, String>
}


impl MultiClassLogistic {

    pub fn new(
        features: &NDArray<f64>, 
        y: &NDArray<f64>,
        activation_function: fn(values: NDArray<f64>) -> NDArray<f64>,
        learning_rate: f64) -> Result<MultiClassLogistic, String> {

        if learning_rate < 0.0 || learning_rate > 1.0 {
            return Err("Learning rate must be between 1 and 0".to_string());
        }

        if y.shape().dim(1) <= 1 {
            return Err("Outputs must be one hot encoded".to_string());
        }

    
        let weights_shape = vec![
            features.shape().dim(1), 
            y.shape().dim(1)
        ];

        let weights = NDArray::new(weights_shape).unwrap();
        let bias = NDArray::new(vec![1, 1]).unwrap();
        let inputs = Value::new(features); 
        let outputs = Value::new(y);

        Ok(Self {
            features: inputs,
            outputs: outputs,
            weights: Value::new(&weights),
            bias: Value::new(&bias),
            learning_rate: learning_rate,
            activation_function: activation_function, 
            loss_function: categorical_cross_entropy
        })
    }


    pub fn predict(&mut self, inputs: NDArray<f64>) -> NDArray<f64> {

        self.features = Value::new(&inputs); 

        let mut logistic = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );

        logistic.forward();

        let y_pred = apply(
            logistic.value(), 0, 
            self.activation_function
        );

        y_pred.argmax(0)
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
        activation_function: fn(values: NDArray<f64>) -> NDArray<f64>,
        learning_rate: f64) -> std::io::Result<MultiClassLogistic> {

        let weights_file = format!("{}/weights", filepath);
        let bias_path = format!("{}/bias", filepath); 

        let inputs = Value::new(&features); 
        let outputs = Value::new(&y);
        let load_weights = NDArray::load(&weights_file).unwrap();
        let load_bias = NDArray::load(&bias_path).unwrap();

        Ok(MultiClassLogistic {
            features: inputs.clone(),
            outputs: outputs.clone(),
            weights: Value::new(&load_weights),
            bias: Value::new(&load_bias),
            learning_rate: learning_rate,
            activation_function: activation_function,
            loss_function: mse,
        })

    }


    pub fn train(&mut self, epochs: usize, log_output: bool) {

        /* create node graph */     
        let mut logistic = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        );

        for epoch in 0..epochs {

            logistic.forward();

            let y_pred = apply(
                logistic.value(), 0, 
                self.activation_function
            );

            let loss = (self.loss_function)(&y_pred, &self.outputs.val()).unwrap();
            let error = y_pred.subtract(self.outputs.val()).unwrap();
            logistic.backward(error);

            /* update weights */
            let n = y_pred.shape().dim(0);
            let learning_rate_factor = (1.0/n as f64) * self.learning_rate;

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

        let mut logistic = ScaleAdd::new(
            Dot::new(self.features.clone(), self.weights.clone()),
            self.bias.clone()
        ); 

        for epoch in 0..epochs {

            let mut batch_index = 0;
            for batch in &x_train {

                self.features.set_val(&batch);
                self.outputs.set_val(&y_train[batch_index]);

                logistic.forward();

                let y_pred = apply(
                    logistic.value(), 0, 
                    self.activation_function
                );
                loss = (self.loss_function)(&y_pred, &self.outputs.val()).unwrap();
                let error = y_pred.subtract(self.outputs.val()).unwrap();

                logistic.backward(error);

                /* update weights */
                let learning_rate_factor = self.learning_rate/y_pred.size() as f64;

                let w_grad = self.features
                    .grad()
                    .scalar_mult(learning_rate_factor)
                    .unwrap();

                let dw = self.weights.val().subtract(w_grad).unwrap();
                self.weights.set_val(&dw); 

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
