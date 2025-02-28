
#[cfg(test)]
mod linear {

    use ndarray::prelude::*; 
    use ndarray::{arr2}; 
    use dendritic_regression::linear_new::*;

    #[test]
    fn test_linear_model() {

        let X = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

        let linear_model = LinearRegression::new(&X, &y).unwrap();
        assert_eq!(linear_model.X.shape(), &[5, 3]); 
        assert_eq!(linear_model.y.shape(), &[5, 1]); 
        assert_eq!(linear_model.coefficients.shape(), &[3, 1]); 

        let X_bad = arr2(&[
            [1.0, 2.0],
            [1.0, 2.0]
        ]);

        let y_bad = arr2(&[[1.0], [2.0], [3.0]]);

        let bad_model = LinearRegression::new(&X_bad, &y_bad);
        assert_eq!(
            bad_model.unwrap_err(),
            "Rows of X and y must be equal: 2 != 3"
        );
 
    }


    #[test]
    fn test_linear_model_fit() -> Result<(), Box<dyn std::error::Error>> {

        let X = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

        let mut linear_model = LinearRegression::new(&X, &y).unwrap();
        linear_model.fit()?;

        Ok(())
 
    }

    /*
    #[test]
    fn test_linear_model() {

        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs"; 

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let model = Linear::new(
            &x, &y, 0.01
        ).unwrap();

        assert_eq!(model.features.val().shape().values(), x.shape().values()); 
        assert_eq!(model.features.val().rank(), 2); 
        assert_eq!(
            model.features.val().values(), 
            x.values()
        );

        assert_eq!(model.outputs.val().shape().values(), y.shape().values()); 
        assert_eq!(model.outputs.val().rank(), 2); 
        assert_eq!(
            model.outputs.val().values(), 
            y.values()
        );

        let expected_shape = vec![model.features.val().shape().dim(1), 1];
        assert_eq!(model.weights.val().shape().values(), expected_shape); 
        assert_eq!(model.weights.val().rank(), 2);
        for item in model.weights.val().values() {
            let expected: f64 = 0.0;
            assert_eq!(item, &expected); 
        }

        let expected_shape = vec![1, 1];
        assert_eq!(model.bias.val().shape().values(), expected_shape); 
        assert_eq!(model.bias.val().rank(), 2);
        for item in model.bias.val().values() {
            let expected: f64 = 0.0;
            assert_eq!(item, &expected); 
        }
    }

    #[test]
    fn test_linear_train() {
    
        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs"; 

        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let mut model = Linear::new(&x, &y, 0.01).unwrap();

        let weights_binding = model.weights.val(); 
        let bias_binding = model.bias.val();
        let weights_prior = weights_binding.values(); 
        let bias_prior = bias_binding.values(); 

        model.train(1000, false);

        let w_binding = model.weights.val(); 
        let b_binding = model.bias.val(); 
        let weights_after = w_binding.values(); 
        let bias_after = b_binding.values(); 

        assert_ne!(weights_prior, weights_after); 
        assert_ne!(bias_prior, bias_after);

        let results = model.predict(x);
        let loss = mse(&results, &y).unwrap(); 
        let loss_condition = loss < 0.1; 
        assert_eq!(loss_condition, true); 
    }

    #[test]
    fn test_linear_sgd() {

        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs"; 

        let batch_size: usize = 2; 
        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let mut model = Linear::new(
            &x, &y, 0.01
        ).unwrap();

        let weights_binding = model.weights.val(); 
        let bias_binding = model.bias.val();
        let weights_prior = weights_binding.values(); 
        let bias_prior = bias_binding.values(); 

        model.sgd(500, false, 2);

        let w_binding = model.weights.val(); 
        let b_binding = model.bias.val(); 
        let weights_after = w_binding.values(); 
        let bias_after = b_binding.values();

        assert_ne!(weights_prior, weights_after); 
        assert_ne!(bias_prior, bias_after);
        let x_train = x.batch(batch_size).unwrap();
        let y_train = y.batch(batch_size).unwrap();

        let results = model.predict(x_train[1].clone());
        let loss = mse(&results, &y_train[1].clone()).unwrap(); 
        let loss_condition = loss < 0.1; 
        assert_eq!(loss_condition, true); 

    }

    
    #[test]
    fn test_linear_save_load() -> std::io::Result<()> {

        let model_path = "data/models/linear";
        let x_path = "data/linear_modeling_data/inputs";
        let y_path = "data/linear_modeling_data/outputs"; 

        let batch_size: usize = 2; 
        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();

        let mut model = Linear::new(
            &x, &y, 0.01
        ).unwrap();

        model.sgd(500, false, 2);
        model.save(model_path).unwrap();


        let x_train = x.batch(batch_size).unwrap();
        let y_train = y.batch(batch_size).unwrap();

        let mut loaded_model = Linear::load(
            model_path, &x, &y, 0.01
        ).unwrap();

        let results = loaded_model.predict(x_train[1].clone());
        let loss = mse(&results, &y_train[1].clone()).unwrap(); 
        let loss_condition = loss < 0.1; 
        assert_eq!(loss_condition, true); 

        Ok(())

    } */


}
