
#[cfg(test)]
mod random_forest_tests {

    use std::fs; 
    use dendritic_ndarray::ndarray::NDArray;
    use dendritic_ndarray::ops::*;
    use dendritic_trees::decision_tree::*; 
    use dendritic_trees::random_forest::*;
    use dendritic_trees::utils::*; 
    use dendritic_metrics::utils::*;
    use dendritic_metrics::loss::*;

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
    fn test_bootstrap_trees() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap(); 
        let target: NDArray<f64> = NDArray::load(y_path).unwrap(); 

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

        model.bootstrap_trees(&dataset, &target).unwrap();
        assert_eq!(model.trees().len(), 4);

        for tree in model.trees() {
           let root_ft_idx = tree.root().feature_idx();
           assert_eq!(root_ft_idx >= 0, true);
        }

        let mut bad_model = RandomForestClassifier::new(
            3, 3,
            4, 10,
            entropy
        );

        let result = bad_model.bootstrap_trees(&dataset, &target);
        assert_eq!(
            result.unwrap_err(), 
            "Random Forest: Number of bootstrap features too large"
        );

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
    fn test_save_rf_classifier() {

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
        model.save("data/random_forest_classifier").unwrap();
        assert_eq!(model.trees().len(), 100);
 
        let mut counter = 0; 
        let paths = fs::read_dir("data/random_forest_classifier").unwrap();
        for path in paths {
            let expected = format!(
                "data/random_forest_classifier/tree_{}", 
                counter
            );
            let actual = path.unwrap().path().display().to_string();
            assert!(actual.contains("data/random_forest_classifier/tree_"));
            counter += 1; 
        } 
    } 


    #[test]
    fn test_load_rf_classifier() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let target: NDArray<f64> = NDArray::load(y_path).unwrap();
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![12, 4]); 
        assert_eq!(target.shape().values(), vec![12, 1]); 

        let mut save_model = RandomForestClassifier::new(
            3, 3,
            100, 3,
            entropy
        );

        save_model.fit(&dataset, &target);
        save_model.save("data/load_random_forest_classifier").unwrap();

        let mut model = RandomForestClassifier::load(3, 3, entropy); 

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), 0); 
        assert_eq!(model.num_features(), 0);
        assert_eq!(model.trees().len(), 0);

        model.fit_loaded(
            &dataset,
            &target,
            "data/load_random_forest_classifier"
        );

        let predictions = model.predict(dataset.clone());
        assert_eq!(predictions.values(), target.values());
        assert_eq!(
            predictions.shape().values(),
            target.shape().values()
        );

    }

    #[test]
    fn test_frequency_check() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let target: NDArray<f64> = NDArray::load(y_path).unwrap();
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();

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

        let predictions = target.values().clone();
        let val = model.frequency_check(predictions).unwrap();
        assert_eq!(val, 2);

        let equal: Vec<f64> = vec![1.0, 3.0, 3.0, 4.0, 4.0, 3.0]; 
        let val2 = model.frequency_check(equal).unwrap();
        assert_eq!(val2, 3);

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

        let n_trees = 4;
        let x_path = "data/unit_testing/inputs";
        let y_path = "data/unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![14, 2]); 
        assert_eq!(target.shape().values(), vec![14, 1]); 
         
        let mut model = RandomForestRegressor::new(
            3, 3,
            n_trees, 1,
            mse
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), n_trees); 
        assert_eq!(model.num_features(), 1);
        assert_eq!(model.trees().len(), 0); 

        model.fit(&dataset, &target);
        assert_eq!(model.trees().len(), n_trees);
        assert_eq!(model.n_trees(), n_trees); 
    }

    #[test]
    fn test_predict_random_forest_regressor() {

        let x_path = "data/linear_unit_testing/inputs";
        let y_path = "data/linear_unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![5, 4]); 
        assert_eq!(target.shape().values(), vec![5, 1]); 
 
        let n_trees = 500; 
        let mut model = RandomForestRegressor::new(
            3, 3,
            n_trees, 3,
            mse
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), n_trees); 
        assert_eq!(model.num_features(), 3);
        assert_eq!(model.trees().len(), 0); 

        model.fit(&dataset, &target);
        assert_eq!(model.trees().len(), n_trees);
        assert_eq!(model.n_trees(), n_trees); 

        let mut idx = 0; 
        let predictions = model.predict(dataset);

        for pred in predictions.values() {
            let diff = pred-target.values()[idx];
            assert_eq!(diff < 3.0, true); 
            idx += 1; 
        } 
 
    }

    #[test]
    fn test_save_random_forest_regressor() {

        let n_trees = 500; 
        let x_path = "data/linear_unit_testing/inputs";
        let y_path = "data/linear_unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![5, 4]); 
        assert_eq!(target.shape().values(), vec![5, 1]);

        let mut model = RandomForestRegressor::new(
            3, 3,
            n_trees, 3,
            mse
        );
 
        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), n_trees); 
        assert_eq!(model.num_features(), 3);
        assert_eq!(model.trees().len(), 0); 

        model.fit(&dataset, &target);
        model.save("data/random_forest_regressor").unwrap();
        assert_eq!(model.trees().len(), n_trees);

        let mut counter = 0; 
        let paths = fs::read_dir("data/random_forest_regressor").unwrap();
        for path in paths {
            let expected = format!(
                "data/random_forest_regressor/tree_{}", 
                counter
            );
            let actual = path.unwrap().path().display().to_string();
            assert!(actual.contains("data/random_forest_regressor/tree_"));
            counter += 1; 
        } 

    }


    #[test]
    fn test_bootstrap_regression_trees() {

        let n_trees = 500; 
        let x_path = "data/linear_unit_testing/inputs";
        let y_path = "data/linear_unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![5, 4]); 
        assert_eq!(target.shape().values(), vec![5, 1]);

        let mut model = RandomForestRegressor::new(
            3, 3,
            n_trees, 3,
            mse
        );
 
        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), n_trees); 
        assert_eq!(model.num_features(), 3);
        assert_eq!(model.trees().len(), 0); 

        model.bootstrap_trees(&dataset, &target).unwrap();
        assert_eq!(model.trees().len(), n_trees);
        assert_eq!(model.n_trees(), n_trees); 

        for tree in model.trees() {
           let root_ft_idx = tree.root().feature_idx();
           assert_eq!(root_ft_idx >= 0, true);
        }

        let mut bad_model = RandomForestRegressor::new(
            3, 3,
            n_trees, 10,
            mse
        );

        let result = bad_model.bootstrap_trees(&dataset, &target);
        assert_eq!(
            result.unwrap_err(), 
            "Random Forest: Number of bootstrap features too large"
        );

    }


    #[test]
    fn test_load_random_forest_regressor() {

        let n_trees = 500; 
        let x_path = "data/linear_unit_testing/inputs";
        let y_path = "data/linear_unit_testing/outputs";
        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        assert_eq!(dataset.shape().values(), vec![5, 4]); 
        assert_eq!(target.shape().values(), vec![5, 1]);

        let mut model = RandomForestRegressor::new(
            3, 3,
            n_trees, 3,
            mse
        );

        assert_eq!(model.max_depth(), 3); 
        assert_eq!(model.samples_split(), 3); 
        assert_eq!(model.n_trees(), n_trees); 
        assert_eq!(model.num_features(), 3);
        assert_eq!(model.trees().len(), 0); 

        model.fit(&dataset, &target);
        model.save("data/load_random_forest_regressor").unwrap();

        let mut loaded_model = RandomForestRegressor::load(3, 3, mse);

        assert_eq!(loaded_model.max_depth(), 3); 
        assert_eq!(loaded_model.samples_split(), 3); 
        assert_eq!(loaded_model.n_trees(), 0); 
        assert_eq!(loaded_model.num_features(), 0);
        assert_eq!(loaded_model.trees().len(), 0); 

        loaded_model.fit_loaded(
            &dataset,
            &target,
            "data/load_random_forest_regressor"
        );

        let mut idx = 0; 
        let predictions = model.predict(dataset);

        for pred in predictions.values() {
            let diff = pred-target.values()[idx];
            assert_eq!(diff < 3.0, true); 
            idx += 1; 
        } 



    }




}
