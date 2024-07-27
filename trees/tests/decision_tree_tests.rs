

#[cfg(test)]
mod decision_tree_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use trees::decision_tree::*; 

    #[test]
    fn test_split() {

        let features: NDArray<f64> = NDArray::array(
            vec![8, 2],
            vec![
                69.0, 4.39,
                69.0, 4.21,
                65.0, 4.09,
                72.0, 5.85,
                73.0, 5.68,
                70.0, 5.56,
                73.0, 5.79,
                65.0, 4.27
            ]
        ).unwrap();
 
        let target: NDArray<f64> = NDArray::array(
       vec![8, 1],
            vec![0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0]
        ).unwrap();

        let model = DecisionTreeClassifier::new(2, 3);

        let (left, right) = model.split(features.clone(), 69.0, 0);
        let (left_2, right_2) = model.split(features, 5.56, 1);

        assert_eq!(left.shape().values(), vec![4, 2]);
        assert_eq!(right.shape().values(), vec![4, 2]);

        let left_vals = left.axis(1, 0).unwrap();
        let right_vals = right.axis(1, 0).unwrap();

        assert_eq!(left_vals.values(), &vec![69.0, 69.0, 65.0, 65.0]);
        assert_eq!(
            right_vals.values(),
            &vec![72.0, 73.0, 70.0, 73.0]
        );

        assert_eq!(left_2.shape().values(), vec![5, 2]);
        assert_eq!(right_2.shape().values(), vec![3, 2]);

        let left_vals_2 = left_2.axis(1, 1).unwrap();
        let right_vals_2 = right_2.axis(1, 1).unwrap();

        assert_eq!(
            left_vals_2.values(), 
            &vec![4.39, 4.21, 4.09, 5.56, 4.27]
        );
   
        assert_eq!(
            right_vals_2.values(), 
            &vec![5.85, 5.68, 5.79]
        );

    }

    #[test]
    fn test_information_gain() {

        let dataset: NDArray<f64> = NDArray::array(
            vec![12, 3],
            vec![
                69.0, 4.39, 0.0,
                69.0, 4.21, 0.0,
                65.0, 4.09, 0.0,
                72.0, 5.85, 1.0,
                73.0, 5.68, 1.0,
                70.0, 5.56, 1.0,
                73.0, 5.79, 1.0,
                65.0, 4.27, 0.0,
                72.0, 6.60, 2.0,
                74.0, 6.75, 2.0,
                71.0, 6.69, 2.0,
                73.0, 6.71, 2.0
            ]
        ).unwrap();

        let target: NDArray<f64> = NDArray::array(
       vec![8, 1],
            vec![0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0]
        ).unwrap();
 
        let model = DecisionTreeClassifier::new(2, 3);

        let (left, right) = model.split(dataset.clone(), 69.0, 0);
        let (left_2, right_2) = model.split(dataset.clone(), 72.0, 0);
        let (left_3, right_3) = model.split(dataset.clone(), 65.0, 0);

        assert_eq!(left.shape().values(), vec![4, 3]); 
        assert_eq!(right.shape().values(), vec![8, 3]); 

        assert_eq!(left_2.shape().values(), vec![8, 3]); 
        assert_eq!(right_2.shape().values(), vec![4, 3]); 

        assert_eq!(left_3.shape().values(), vec![2, 3]); 
        assert_eq!(right_3.shape().values(), vec![10, 3]); 

        let info_gain = model.information_gain(
            dataset.axis(1, 2).unwrap(), 
            left.axis(1, 2).unwrap(), 
            right.axis(1, 2).unwrap()
        );

        let info_gain_2 = model.information_gain(
            dataset.axis(1, 2).unwrap(),
            left_2.axis(1, 2).unwrap(),
            right_2.axis(1, 2).unwrap()
        ); 

        let info_gain_3 = model.information_gain(
            dataset.axis(1, 2).unwrap(),
            left_3.axis(1, 2).unwrap(),
            right_3.axis(1, 2).unwrap()
        );

        assert_eq!(info_gain < info_gain_2, false);
        assert_eq!(info_gain_3 < info_gain, true);
    }


    #[test]
    fn test_best_split() {

        let dataset: NDArray<f64> = NDArray::array(
            vec![12, 3],
            vec![
                69.0, 4.39, 0.0,
                69.0, 4.21, 0.0,
                65.0, 4.09, 0.0,
                72.0, 5.85, 1.0,
                73.0, 5.68, 1.0,
                70.0, 5.56, 1.0,
                73.0, 5.79, 1.0,
                65.0, 4.27, 0.0,
                72.0, 6.60, 2.0,
                74.0, 6.75, 2.0,
                71.0, 6.69, 2.0,
                73.0, 6.71, 2.0
            ]
        ).unwrap();

        let target: NDArray<f64> = NDArray::array(
            vec![8, 1],
            vec![0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0]
        ).unwrap();

        let model = DecisionTreeClassifier::new(3, 3);

        let (info_gain, feature, threshold) =  model.best_split(dataset);

        assert_eq!(feature, 0); 
        assert_eq!(threshold, 69.0);
        assert_eq!(info_gain, 0.9182958340544894);
    }


    #[test]
    fn test_build_tree() {

        let dataset: NDArray<f64> = NDArray::array(
            vec![12, 3],
            vec![
                69.0, 4.39, 0.0,
                69.0, 4.21, 0.0,
                65.0, 4.09, 0.0,
                72.0, 5.85, 1.0,
                73.0, 5.68, 1.0,
                70.0, 5.56, 1.0,
                73.0, 5.79, 1.0,
                65.0, 4.27, 0.0,
                72.0, 6.60, 2.0,
                74.0, 6.75, 2.0,
                71.0, 6.69, 2.0,
                73.0, 6.71, 2.0
            ]
        ).unwrap();


        let target: NDArray<f64> = NDArray::array(
            vec![12, 1],
            vec![0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,2.0,2.0,2.0,2.0]
        ).unwrap();

        let mut model = DecisionTreeClassifier::new(3, 3);
        model.fit(dataset.clone(), target);
        let predictions = model.predict(dataset);
        let expected = vec![
            0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,2.0,2.0,2.0,2.0
        ];

        assert_eq!(predictions.shape().values(), vec![12, 1]);
        assert_eq!(predictions.values(), &expected);

    }


}


