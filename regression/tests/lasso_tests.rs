
#[cfg(test)]
mod lasso {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use metrics::loss::*;
    use regression::lasso::*;

    #[test]
    fn test_lasso_model_train() {

        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.001; 

        let mut model = Lasso::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.train(1000, false);
        let output = model.predict(x);
        let loss = mse(&y, &output);
        assert_eq!(loss < Ok(0.1), true);  
    }


    #[test]
    fn test_lasso_sgd() {

        let x_path = "data/linear_modeling_data/inputs"; 
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.001; // lower lambda value for batches 
        let expected_predictions: Vec<f64> = vec![
            9.0, 11.0, 13.0, 15.0, 18.0
        ];

        let mut model = Lasso::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.sgd(500, false, 2);
        let output = model.predict(x);
        let loss = mse(&y, &output);
        let condition = output.values() > &expected_predictions;

        assert_eq!(loss < Ok(0.1), true);
        assert_eq!(condition, true); 

    }


    #[test]
    fn test_lasso_save_load() -> std::io::Result<()> {

        let model_path = "data/models/lasso";
        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs"; 

        let batch_size: usize = 2; 
        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.5;  
        let expected_predictions: Vec<f64> = vec![10.0, 13.0];

        let mut model = Lasso::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.sgd(500, false, 2);
        model.save(model_path).unwrap();

        let x_train = x.batch(batch_size).unwrap();
        let y_train = y.batch(batch_size).unwrap();

        let mut loaded_model = Lasso::load(
            model_path,
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        let results = loaded_model.predict(x_train[1].clone());
        let loss = mse(&results, &y_train[1].clone()).unwrap(); 

        assert_eq!(loss < 1.0, true); 
        assert_eq!(results.values() > &expected_predictions, true); 

        Ok(())

    }

}
