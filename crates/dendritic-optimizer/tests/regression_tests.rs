
#[cfg(test)]
mod graph_test {

    use std::fs;
    use std::fs::File;

    use ndarray::arr2;

    use dendritic_optimizer::train::*; 
    use dendritic_optimizer::regression::*;

    #[test]
    fn test_linear() -> std::io::Result<()> {

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

        let mut model = Regression::new(&x, &y, 0.001).unwrap();

        assert_eq!(model.weight_dim, (3, 1));
        assert_eq!(model.bias_dim, (1, 1));
        assert_eq!(model.learning_rate, 0.001);
        assert_eq!(model.input(), x); 
        assert_eq!(model.output(), y); 

        model.train(1000);
        model.save("data/linear")?;

        let mut loaded_model = Regression::load("data/linear").unwrap();
        let output = loaded_model.predict(&x);
        let diff = output - y; 
        assert_eq!(diff.sum() < 0.2, true);

        fs::remove_dir_all("data/linear")?; 
        Ok(())
    }


    #[test]
    fn test_ridge() -> std::io::Result<()> {

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

        let mut model = Ridge::new(&x, &y, 0.001, 0.001).unwrap();

        assert_eq!(model.regression.weight_dim, (3, 1));
        assert_eq!(model.regression.bias_dim, (1, 1));
        assert_eq!(model.regression.learning_rate, 0.001);
        assert_eq!(model.lambda, 0.001);
        assert_eq!(model.regression.input(), x); 
        assert_eq!(model.regression.output(), y); 

        model.train(1000);
        model.save("data/ridge")?;

        let mut loaded_model = Ridge::load("data/ridge").unwrap();
        let output = loaded_model.predict(&x);
        let diff = output - y; 
        assert_eq!(diff.sum() < 0.2, true);

        fs::remove_dir_all("data/ridge")?; 
        Ok(())
    }


    #[test]
    fn test_lasso() -> std::io::Result<()> {

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

        let mut model = Lasso::new(&x, &y, 0.001, 0.001).unwrap();

        assert_eq!(model.regression.weight_dim, (3, 1));
        assert_eq!(model.regression.bias_dim, (1, 1));
        assert_eq!(model.regression.learning_rate, 0.001);
        assert_eq!(model.lambda, 0.001);
        assert_eq!(model.regression.input(), x); 
        assert_eq!(model.regression.output(), y); 

        model.train(1000);
        model.save("data/lasso")?;

        let mut loaded_model = Ridge::load("data/lasso").unwrap();
        let output = loaded_model.predict(&x);
        let diff = output - y; 
        assert_eq!(diff.sum() < 0.2, true);

        fs::remove_dir_all("data/lasso")?; 
        Ok(())
    }

    #[test]
    fn test_elastic() -> std::io::Result<()> {

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

        let mut model = Elastic::new(&x, &y, 0.001, 0.001, 0.5).unwrap();

        assert_eq!(model.regression.weight_dim, (3, 1));
        assert_eq!(model.regression.bias_dim, (1, 1));
        assert_eq!(model.regression.learning_rate, 0.001);
        assert_eq!(model.lambda, 0.001);
        assert_eq!(model.alpha, 0.5);
        assert_eq!(model.regression.input(), x); 
        assert_eq!(model.regression.output(), y); 

        model.train(1000);
        model.save("data/elastic")?;

        let mut loaded_model = Ridge::load("data/elastic").unwrap();
        let output = loaded_model.predict(&x);
        let diff = output - y; 
        assert_eq!(diff.sum() < 0.2, true);

        fs::remove_dir_all("data/elastic")?; 
        Ok(())
    }

}
