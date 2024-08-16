
#[cfg(test)]
mod random_forest_tests {

    use std::fs; 
    use ndarray::ndarray::NDArray;
    use ndarray::ops::*; 
    use trees::random_forest::*;
    use trees::utils::*; 
    use metrics::utils::*;
    use metrics::loss::*;

    #[test]
    fn test_random_forest_classifier() {

        let model = RandomForestClassifier::new(
            3, 3,
            4, 2,
            entropy
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), 4); 
        assert_eq!(model.num_features(), 2);
        assert_eq!(model.trees().len(), 0); 

    }

    #[test]
    fn test_fit_random_forest_classifier() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap(); 
        let target: NDArray<f64> = NDArray::load(y_path).unwrap(); 

        assert_eq!(dataset.shape().values(), vec![12, 4]); 
        assert_eq!(target.shape().values(), vec![12, 1]); 

        let mut model = RandomForestClassifier::new(
            3, 3,
            4, 2,
            entropy
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), 4); 
        assert_eq!(model.num_features(), 2);
        assert_eq!(model.trees().len(), 0); 
        assert_eq!(model.trees().len(), 0); 

        model.fit(&dataset, &target);

        assert_eq!(model.trees().len(), 4);

    }

    #[test]
    fn test_predict_random_forest_classifier() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let target: NDArray<f64> = NDArray::load(y_path).unwrap();
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap(); 

        assert_eq!(dataset.shape().values(), vec![12, 4]); 
        assert_eq!(target.shape().values(), vec![12, 1]); 

        let mut model = RandomForestClassifier::new(
            3, 3,
            100, 3,
            entropy
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), 100); 
        assert_eq!(model.num_features(), 3);
        assert_eq!(model.trees().len(), 0); 
 
        model.fit(&dataset, &target);

        assert_eq!(model.trees().len(), 100); 

        let predictions = model.predict(dataset);

        assert_eq!(
            predictions.values().to_vec(),
            target.values().to_vec()
        ); 
    }


    #[test]
    fn test_save_load_rf_regressor() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let target: NDArray<f64> = NDArray::load(y_path).unwrap();
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![12, 4]); 
        assert_eq!(target.shape().values(), vec![12, 1]); 

        let mut model = RandomForestClassifier::new(
            3, 3,
            100, 3,
            entropy
        );
 
        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), 100); 
        assert_eq!(model.num_features(), 3);
        assert_eq!(model.trees().len(), 0); 

        model.fit(&dataset, &target);
        model.save("data/rf_classifier").unwrap();
        assert_eq!(model.trees().len(), 100);

        let mut counter = 0; 
        let paths = fs::read_dir("data/rf_classifier").unwrap();
        for path in paths {
            let expected = format!(
                "data/rf_classifier/tree_{}.json", 
                counter
            );
            let actual = path.unwrap().path().display().to_string();
            assert!(actual.contains("data/rf_classifier/tree_"));
            assert!(actual.ends_with(".json"));
            counter += 1; 
        }

    }


    #[test]
    fn test_random_forest_regressor() {
         
        let model = RandomForestRegressor::new(
            3, 3,
            4, 2,
            mse
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), 4); 
        assert_eq!(model.num_features(), 2);
        assert_eq!(model.trees().len(), 0); 
    }

    #[test]
    fn test_fit_random_forest_regressor() {

        let x_path = "data/unit_testing/inputs";
        let y_path = "data/unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![14, 2]); 
        assert_eq!(target.shape().values(), vec![14, 1]); 
         
        let mut model = RandomForestRegressor::new(
            3, 3,
            4, 1,
            mse
        );

        model.fit(&dataset, &target);
        assert_eq!(model.trees().len(), 4); 
    }

    #[test]
    fn test_predict_random_forest_regressor() {

        let x_path = "data/linear_unit_testing/inputs";
        let y_path = "data/linear_unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![5, 4]); 
        assert_eq!(target.shape().values(), vec![5, 1]); 
 
        let n_trees = 100; 
        let mut model = RandomForestRegressor::new(
            3, 3,
            n_trees, 2,
            mse
        );

        model.fit(&dataset, &target);
        assert_eq!(model.trees().len(), n_trees);

        let mut idx = 0; 
        let predictions = model.predict(dataset);
        for pred in predictions.values() {
            let diff = pred-target.values()[idx];
            assert_eq!(diff < 2.0, true); 
            idx += 1; 
        } 
 
    }


}
