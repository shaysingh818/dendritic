
#[cfg(test)]
mod classification_test {

    use std::fs;
    use std::fs::File;

    use ndarray::{arr2, Array1};

    use dendritic_optimizer::train::*;
    use dendritic_optimizer::regression::*; 
    use dendritic_optimizer::classification::*;
 
    #[test]
    fn test_binary_classification() -> std::io::Result<()> {

        // binary logstic data
        let x = arr2(&[
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.8],
            [3.0, 3.2],
            [2.8, 3.0],
            [5.0, 5.5],
            [6.0, 5.8],
            [5.5, 6.0],
            [6.2, 5.9],
            [7.0, 6.5]
        ]);

        let y = arr2(&[
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0]
        ]);

        let mut model = Logistic::new(&x, &y, false, 0.01).unwrap();

        assert_eq!(model.weight_dim, (2, 1));
        assert_eq!(model.bias_dim, (1, 1));
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.input(), x); 
        assert_eq!(model.output(), y); 

        model.train(2000);
        model.save("data/binary_logistic")?;

        let mut loaded_model = Logistic::load(
            "data/binary_logistic"
        ).unwrap();
        let output = loaded_model.predict(&x);
        let diff = output - y; 
        assert_eq!(diff.sum() < 0.2, true);

        fs::remove_dir_all("data/binary_logistic")?; 
        Ok(())
    } 


    #[test]
    fn test_multi_classification() -> std::io::Result<()> {

        // multi class
        let x1 = arr2(&[
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 1.0],   // Class 0
            [4.0, 4.5],
            [4.5, 4.8],
            [5.0, 5.2],   // Class 1
            [7.0, 7.5],
            [7.5, 8.0],
            [8.0, 8.5],   // Class 2
        ]);

        let y1 = arr2(&[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]);

        let expected: Array1<f64> = Array1::from(vec![
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
        ]);


        let mut model = Logistic::new(&x1, &y1, true, 0.01).unwrap();

        assert_eq!(model.weight_dim, (2, 3));
        assert_eq!(model.bias_dim, (1, 1));
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.input(), x1); 
        assert_eq!(model.output(), y1); 

        model.train(2000);
        model.save("data/multiclass_logistic")?;

        let mut loaded = Logistic::load(
            "data/multiclass_logistic"
        ).unwrap(); 
        let output = loaded.predict(&x1);
        let class_predictions = output.column(0); 
        assert_eq!(class_predictions, expected); 
        Ok(())
    }

}
