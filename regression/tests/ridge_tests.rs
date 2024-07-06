
#[cfg(test)]
mod ridge_test {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use autodiff::node::*; 
    use metrics::mse::*;
    use regression::ridge::*;

    #[test]
    fn test_ridge_model_train() {

        let x_path = "data/linear_modeling_data/inputs"; 
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.0001; 

        let mut model = Ridge::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.train(1100, false);
        let output = model.predict(x);
        let loss = mse(&y, &output);

        assert_eq!(loss < Ok(0.1), true);  
    }


    #[test]
    fn test_ridge_sgd() {

        let x_path = "data/linear_modeling_data/inputs"; 
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.01; // lower lambda value for batches 
        let expected_predictions: Vec<f64> = vec![
            9.0, 11.0, 13.0, 15.0, 18.0
        ];

        let mut model = Ridge::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.sgd(1100, false, 2);
        let output = model.predict(x);
        let loss = mse(&y, &output);
        let condition = output.values() > &expected_predictions;

        assert_eq!(loss < Ok(0.1), true);
        assert_eq!(condition, true); 

    }


    #[test]
    fn test_ridge_weight_regularization() {

        let x_path = "data/linear_modeling_data/inputs"; 
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambdas = vec![0.0001, 1.0, 10.0, 100.0, 200.0];
        let mut weights: Vec<NDArray<f64>> = Vec::new();
        let mut avgs: Vec<f64> = Vec::new();

        for lambda in &lambdas {

            let mut model = Ridge::new(
                x.clone(), y.clone(),
                *lambda, learning_rate
            ).unwrap();

            model.train(1100, false);
            let _output = model.predict(x.clone());
            weights.push(model.weights.value());
        }

        for weight in &weights {
            let sum: f64 = weight.values().iter().sum();
            let avg: f64 = sum/weight.size() as f64;
            avgs.push(avg);
        }

        /* confirm later weights are less than prior */
        assert_eq!(avgs[0] > avgs[4], true); 
        assert_eq!(avgs[2] > avgs[4], true); 
        assert_eq!(avgs[1] > avgs[3], true); 
    }

    #[test]
    fn test_save_load_ridge_model() {

        let model_path = "data/models/ridge";
        let x_path = "data/linear_modeling_data/inputs"; 
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.5; // lower lambda value for batches 
        let expected_predictions: Vec<f64> = vec![
            7.0, 10.0, 12.0, 15.0, 18.0
        ];

        let mut model = Ridge::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.sgd(1100, false, 2);
        let output = model.predict(x.clone());
        let _loss = mse(&y, &output);
        let condition = output.values() > &expected_predictions;
        assert_eq!(condition, true);

        model.save(model_path).unwrap();

        let mut loaded_model = Ridge::load(
            model_path, x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        let loaded_output = loaded_model.predict(x);
        let l_condition = loaded_output.values() > &expected_predictions;
        assert_eq!(l_condition, true);

    }  

}
