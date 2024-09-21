#[cfg(test)]
mod gaussian_bayes_tests {

    use metrics::loss::*; 
    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use bayes::gaussian_bayes::*;
    use bayes::shared::*; 

    #[test]
    fn test_gaussian_bayes() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();
 
        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
            &target
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![14, 2]); 
        assert_eq!(clf.outputs.shape().values(), vec![14, 1]);

        let bad: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![1.0, 2.0]
        ).unwrap();

        let mut clf_bad = GaussianNB::new(
            &features,
            &bad
        );

        assert_eq!(
            clf_bad.unwrap_err(), 
            "Feature rows must match output rows"
        ); 

    }


    #[test]
    fn test_build_likelihoods() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
    &target
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![14, 2]); 
        assert_eq!(clf.outputs.shape().values(), vec![14, 1]);

        let results = clf.likelihoods(); 
        let class1 = results.axis(1, 0).unwrap(); 
        let class2 = results.axis(1, 1).unwrap();

        assert_eq!(
            *class1.values(),
            vec![
                2.8000000000000003, 
                0.8164965809277259, 
                67.0, 
                6.454972243679028
            ]
        );


        assert_eq!(
            *class2.values(),
            vec![
                1.5, 
                0.6110100926607787, 
                66.0, 
                4.58257569495584
            ]
        );

    }

    #[test]
    fn test_gaussian_distribution() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
            &target
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![14, 2]); 
        assert_eq!(clf.outputs.shape().values(), vec![14, 1]);

        let results = clf.likelihoods(); 
        let class1 = results.axis(1, 0).unwrap(); 
        let class2 = results.axis(1, 1).unwrap();

        let gaussian1 = gaussian_probability(
            2.6, 
            class1.values()[0],
            class1.values()[1]
        );

        let gaussian2 = gaussian_probability(
            2.6, 
            class2.values()[0],
            class2.values()[1]
        );

        assert_eq!(gaussian1, 0.47416212535677055);
        assert_eq!(gaussian2, 0.12914332487788097);
        assert_eq!(gaussian1 > gaussian2, true); 
    }


    #[test]
    fn test_predict_feature() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
            &target
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![14, 2]); 
        assert_eq!(clf.outputs.shape().values(), vec![14, 1]);

        let predict1 = clf.predict_feature(0, 2.6, 0.0);
        let predict2 = clf.predict_feature(0, 2.6, 1.0);
        let predict3 = clf.predict_feature(1, 70.0, 0.0);
        let predict4 = clf.predict_feature(1, 70.0, 1.0);

        assert_eq!(predict1, 0.47416212535677055);
        assert_eq!(predict2, 0.12914332487788097);
        assert_eq!(predict3, 0.0554768613640256);
        assert_eq!(predict4, 0.05947780073027187); 
    }


    #[test]
    fn test_fit_row() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
            &target
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![14, 2]); 
        assert_eq!(clf.outputs.shape().values(), vec![14, 1]);

        let row1: NDArray<f64> = NDArray::array(
            vec![1, 2],
            vec![0.8, 59.0]
        ).unwrap();

        let mut preds: Vec<f64> = Vec::new();
        let rows = features.shape().dim(0); 
        for row in 0..rows {
            let item = features.axis(0, row).unwrap();
            let pred = clf.fit_row(item).unwrap();
            preds.push(pred);
        }

        let predictions = NDArray::array(
            vec![preds.len(), 1],
            preds
        ).unwrap();

        let error = mse(&target, &predictions).unwrap();
        assert_eq!(error, 0.21428571428571427);

        let bad_row1: NDArray<f64> = NDArray::array(
            vec![1, 2],
            vec![0.8, 59.0]
        ).unwrap();

        let pred2 = clf.fit_row(bad_row1); 
        assert_eq!(
            pred2.unwrap_err(),
            "row sample not equal to features column count"
        );
    }


    #[test]
    fn test_fit() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
            &target
        ).unwrap();

        assert_eq!(clf.features.shape().values(), vec![14, 2]); 
        assert_eq!(clf.outputs.shape().values(), vec![14, 1]);

        let preds = clf.fit(features).unwrap();
        let error = mse(&target, &preds).unwrap();
        assert_eq!(error, 0.21428571428571427);

        let bad_row1: NDArray<f64> = NDArray::array(
            vec![2, 1],
            vec![0.8, 59.0]
        ).unwrap();

        let pred2 = clf.fit(bad_row1); 
        assert_eq!(
            pred2.unwrap_err(),
            "row sample not equal to features column count"
        );

    }


    #[test]
    fn test_save_load() {

        let x_path = "data/prostate_cancer/inputs";
        let y_path = "data/prostate_cancer/outputs";

        let features = NDArray::load(x_path).unwrap();
        let target = NDArray::load(y_path).unwrap();

        assert_eq!(features.shape().values(), vec![14, 2]);
        assert_eq!(target.shape().values(), vec![14, 1]);

        let mut clf = GaussianNB::new(
            &features,
            &target
        ).unwrap();

        clf.save("data/models/weather").unwrap(); 

        let mut clf2 = GaussianNB::load(
            "data/models/weather",
            &features,
            &target
        ).unwrap();

        let preds = clf2.fit(features).unwrap();
        let error = mse(&target, &preds).unwrap();
        assert_eq!(error, 0.21428571428571427);

    }    


}

