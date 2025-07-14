
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

        let mut model = Regression::new(&x, &y, 0.008).unwrap();

        assert_eq!(model.weight_dim, (3, 1));
        assert_eq!(model.bias_dim, (1, 1));
        assert_eq!(model.learning_rate, 0.008);
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

}
