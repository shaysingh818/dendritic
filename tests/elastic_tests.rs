use steepgrad::regression;
use steepgrad::ndarray;
use steepgrad::loss;


#[cfg(test)]
mod elastic {

    use crate::ndarray::ndarray::NDArray;
    use crate::ndarray::ops::*;
    use crate::loss::mse::*;
    use crate::regression::elastic_net::*;

    #[test]
    fn test_elastic_net_model_train() {

        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.001; 
        let expected_predictions: Vec<f64> = vec![
            9.0, 11.0, 13.0, 15.0, 18.0
        ];

        let mut model = ElasticNet::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.train(1000, false);
        let output = model.predict(x);
        let loss = mse(&y, &output);
        let condition = output.values() > &expected_predictions;

        assert_eq!(loss < Ok(0.1), true); 
        assert_eq!(condition, true); 
    }


    #[test]
    fn test_elastic_sgd() {

        let batch_size = 2; 
        let x_path = "data/linear_modeling_data/inputs"; 
        let y_path = "data/linear_modeling_data/outputs";

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.0001; // lower lambda value for batches 
        let expected_predictions: Vec<f64> = vec![11.0, 13.0];

        let mut model = ElasticNet::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();


        let x_train = x.batch(batch_size).unwrap();
        let y_train = y.batch(batch_size).unwrap();

        model.sgd(500, false, batch_size);
        let output = model.predict(x_train[1].clone());
        let loss = mse(&output, &y_train[1].clone());
        let condition = output.values() > &expected_predictions;

        assert_eq!(loss < Ok(0.1), true);
        assert_eq!(condition, true); 

    }


    #[test]
    fn test_elastic_load_save() {

        let model_path = "data/models/lasso";
        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs"; 

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let learning_rate = 0.01;
        let lambda = 0.0001;
        let expected_predictions: Vec<f64> = vec![
            9.0, 11.0, 13.0, 15.0, 18.0
        ];

        let mut model = ElasticNet::new(
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        model.train(1000, false);
        model.save(model_path).unwrap();

        let mut loaded_model = ElasticNet::load(
            model_path,
            x.clone(), y.clone(),
            lambda, learning_rate
        ).unwrap();

        let results = loaded_model.predict(x.clone());
        let loss = mse(&results, &y.clone()).unwrap(); 
        let condition = results.values() > &expected_predictions;

        assert_eq!(loss < 0.1, true);
        assert_eq!(condition, true); 


    } 

}
