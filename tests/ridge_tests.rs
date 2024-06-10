use steepgrad::loss; 
use steepgrad::ndarray;
use steepgrad::autodiff;
use steepgrad::regression;

#[cfg(test)]
mod ridge_test {

    use crate::ndarray::ndarray::NDArray;
    use crate::ndarray::ops::*;
    use crate::autodiff::node::*; 
    use crate::autodiff::ops::*; 
    use crate::loss::mse::*;
    use crate::regression::ridge::*;


    #[test]
    fn test_ridge_model_train() {

        let x_path = "data/ridge_testing_data/inputs"; 
        let y_path = "data/ridge_testing_data/outputs";
        let w_path = "data/ridge_testing_data/weights_model";
        let b_path = "data/ridge_testing_data/bias";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let w: NDArray<f64> = NDArray::load(w_path).unwrap();
        let b: NDArray<f64> = NDArray::load(b_path).unwrap();

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

        let x_path = "data/ridge_testing_data/inputs"; 
        let y_path = "data/ridge_testing_data/outputs";
        let w_path = "data/ridge_testing_data/weights_model";
        let b_path = "data/ridge_testing_data/bias";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let w: NDArray<f64> = NDArray::load(w_path).unwrap();
        let b: NDArray<f64> = NDArray::load(b_path).unwrap();

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

        let x_path = "data/ridge_testing_data/inputs"; 
        let y_path = "data/ridge_testing_data/outputs";
        let w_path = "data/ridge_testing_data/weights_model";
        let b_path = "data/ridge_testing_data/bias";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let w: NDArray<f64> = NDArray::load(w_path).unwrap();
        let b: NDArray<f64> = NDArray::load(b_path).unwrap();

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
            let output = model.predict(x.clone());
            let loss = mse(&y, &output);
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

}
