
#[cfg(test)]
mod knn_tests {

    use dendritic_ndarray::ndarray::NDArray;
    use dendritic_ndarray::ops::*;
    use dendritic_knn::knn::*;
    use dendritic_knn::distance::*;
    use dendritic_knn::utils::*;

    #[test]
    fn test_knn_classifier() {

        let x_path = "data/knn_sample/inputs";
        let y_path = "data/knn_sample/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![7, 2]);
        assert_eq!(target.shape().values(), vec![7, 1]);

        let k = 7;
        let clf = KNN::fit(&features, &target, k, euclidean).unwrap();

        assert_eq!(clf.features.shape().values(), vec![7, 2]);
        assert_eq!(clf.outputs.shape().values(), vec![7, 1]);
        assert_eq!(clf.k, 7); 

        let bad_features: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![20.0, 40.0]
        ).unwrap();

        let clf_bad = KNN::fit(&bad_features, &target, k, euclidean);
        assert_eq!(
            clf_bad.unwrap_err(),
            "Feature rows must match output rows"
        );

    }

    #[test]
    fn test_calculate_distances() {

        let x_path = "data/knn_sample/inputs";
        let y_path = "data/knn_sample/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![7, 2]);
        assert_eq!(target.shape().values(), vec![7, 1]);

        let test_point: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![20.0, 35.0]
        ).unwrap();

        let bad_point: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![20.0, 35.0, 10.0]
        ).unwrap();

        let k = 5;
        let _clf = KNN::fit(&features, &target, k, euclidean).unwrap();
        let distances = calculate_distances(
            euclidean,
            &features,
            &test_point
        ).unwrap();

        let expected: Vec<(f64, usize)> = vec![
            (14.142135623730951, 3),
            (25.0, 0),
            (33.54101966249684, 1),
            (45.27692569068709, 6),
            (47.16990566028302, 5),
            (61.032778078668514, 4),
            (68.00735254367721, 2)
        ];

        assert_eq!(distances.len(), 7);
        assert_eq!(distances, expected);

        let distance_err = calculate_distances(
            euclidean, 
            &features, 
            &bad_point
        );

        assert_eq!(
            distance_err.unwrap_err(),
            "KNN: Rows of point doesn't match cols of sample data"
        );
    }

    #[test]
    fn test_predict_sample() {

        let x_path = "data/knn_sample/inputs";
        let y_path = "data/knn_sample/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![7, 2]);
        assert_eq!(target.shape().values(), vec![7, 1]);

        let test_point: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![20.0, 35.0]
        ).unwrap();

        let k = 5;
        let clf = KNN::fit(&features, &target, k, euclidean).unwrap();
        let pred = clf.predict_sample(&test_point);
        assert_eq!(pred, 0.0); 
    }


    #[test]
    fn test_predict() {

        let x_path = "data/knn_sample/inputs";
        let y_path = "data/knn_sample/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![7, 2]);
        assert_eq!(target.shape().values(), vec![7, 1]);

        let test_point: NDArray<f64> = NDArray::array(
            vec![3, 2],
            vec![
                20.0, 35.0,
                61.0, 92.0, 
                21.0, 36.0
            ]
        ).unwrap();

        let k = 5;
        let clf = KNN::fit(&features, &target, k, euclidean).unwrap();
        let preds = clf.predict(&test_point);

        assert_eq!(preds.shape().values(), vec![3, 1]);
        assert_eq!(
            preds.values(),
            &vec![0.0, 1.0, 0.0]
        ); 
    }


    #[test]
    fn test_knn_regressor() {

        let x_path = "data/knn_regression/inputs";
        let y_path = "data/knn_regression/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![10, 2]);
        assert_eq!(target.shape().values(), vec![10, 1]);

        let k = 7;
        let clf = KNNRegressor::fit(
            &features, 
            &target, 
            k, 
            euclidean
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![10, 2]);
        assert_eq!(clf.outputs.shape().values(), vec![10, 1]);
        assert_eq!(clf.k, 7); 

        let bad_features: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![20.0, 40.0]
        ).unwrap();

        let clf_bad = KNN::fit(&bad_features, &target, k, euclidean);
        assert_eq!(
            clf_bad.unwrap_err(),
            "Feature rows must match output rows"
        );

    }


    #[test]
    fn test_predict_sample_regression() {

        let x_path = "data/knn_regression/inputs";
        let y_path = "data/knn_regression/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![10, 2]);
        assert_eq!(target.shape().values(), vec![10, 1]);

        let test_point: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![4.5, 7.5]
        ).unwrap();

        let k = 3;
        let clf = KNNRegressor::fit(
            &features, 
            &target, 
            k, 
            euclidean
        ).unwrap();

        let pred = clf.predict_sample(&test_point);
        assert_eq!(pred, 9.166666666666666);
    }

    
    #[test]
    fn test_predict_regression() {

        let x_path = "data/knn_regression/inputs";
        let y_path = "data/knn_regression/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![10, 2]);
        assert_eq!(target.shape().values(), vec![10, 1]);

        let test_point: NDArray<f64> = NDArray::array(
            vec![3, 2],
            vec![
                4.5, 7.5,
                3.0, 5.0, 
                4.0, 7.0
            ]
        ).unwrap();

        let k = 3;
        let clf = KNNRegressor::fit(
            &features, 
            &target, 
            k, 
            euclidean
        ).unwrap();

        let preds = clf.predict(&test_point);

        assert_eq!(preds.shape().values(), vec![3, 1]);
        assert_eq!(
            preds.values(),
            &vec![
                9.1666666666666666, 
                6.0, 
                7.5
            ]
        ); 
    }


}
