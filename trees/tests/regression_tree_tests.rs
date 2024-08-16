
#[cfg(test)]
mod regression_tree_tests {

    use std::fs; 
    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use trees::utils::*; 
    use trees::decision_tree_regressor::*; 
    use metrics::loss::*;
    use datasets::airfoil_noise::*;


    #[test]
    fn test_decision_tree_regressor() {

        let x_path = "data/unit_testing/inputs";
        let features: NDArray<f64> = NDArray::load(x_path).unwrap();
        let (left, right) = split(features, 5.5, 0);

        assert_eq!(left.shape().values(), vec![5, 2]);
        assert_eq!(right.shape().values(), vec![9, 2]);
 
        let left_vals = left.axis(1, 0).unwrap();
        let right_vals = right.axis(1, 0).unwrap();

        assert_eq!(
            left_vals.values(),
            &vec![5.0, 4.0, 3.0, 2.0, 1.0]
        );

        assert_eq!(
            right_vals.values(),
            &vec![
                14.0, 13.0, 12.0, 11.0, 10.0,
                9.0, 8.0, 7.0, 6.0
            ]
        ); 

    }


    #[test]
    fn test_best_split() {

        let x_path = "data/unit_testing/inputs";
        let features: NDArray<f64> = NDArray::load(x_path).unwrap();
        let model = DecisionTreeRegressor::new(3, 3, mse);
        let (mse, feature_idx, threshold) = model.best_split(features);

        assert_eq!(feature_idx, 0);
        assert_eq!(threshold, 5.0); 
        assert_eq!(mse, 2.1389432098765435); 
    }


    #[test]
    fn test_gain() {

        let x_path = "data/unit_testing/inputs";
        let features: NDArray<f64> = NDArray::load(x_path).unwrap();

        let model = DecisionTreeRegressor::new(3, 3, mse);
        let (left, right) = split(features, 5.5, 0);
        let gain = model.gain(
            left.axis(1, 1).unwrap(),
            right.axis(1, 1).unwrap()
        );

        assert_eq!(gain, 2.1389432098765435); 

    }


    #[test]
    fn test_fit_and_predict() {

        let x_path = "data/unit_testing/inputs";
        let y_path = "data/unit_testing/outputs";

        let features: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        let mut model = DecisionTreeRegressor::new(3, 3, mse);
        model.fit(&features, &target);

        let inputs = features.axis(1, 0).unwrap();
        let predictions = model.predict(inputs);

        let diff = target.subtract(predictions).unwrap();
        for item in diff.values() {
            assert_eq!(item < &1.0, true);
        }

    }


    #[test]
    fn test_save_tree() {

        let x_path = "data/unit_testing/inputs";
        let y_path = "data/unit_testing/outputs";
        let save_path = "data/regression_tree";

        let features: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        let mut model = DecisionTreeRegressor::new(3, 3, mse);
        model.fit(&features, &target);
        model.save(save_path).unwrap();

        assert_eq!(fs::metadata(save_path).is_ok(), true);
    }


    #[test]
    fn test_load_tree() {

        let x_path = "data/unit_testing/inputs";
        let y_path = "data/unit_testing/outputs";
        let save_path = "data/regression_tree";

        let features: NDArray<f64> = NDArray::load(x_path).unwrap();
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        let mut model = DecisionTreeRegressor::load(
            save_path,
            3, 3, 
            mse
        );

        model.fit(&features, &target);

        let inputs = features.axis(1, 0).unwrap();
        let predictions = model.predict(inputs);

        let diff = target.subtract(predictions).unwrap();
        for item in diff.values() {
            assert_eq!(item < &1.0, true);
        }

    }

}
