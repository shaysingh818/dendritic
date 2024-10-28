
#[cfg(test)]
mod loss_tests {

    use dendritic_ndarray::ndarray::NDArray;
    use dendritic_metrics::loss::*;
    use dendritic_metrics::utils::*; 


    #[test]
    fn test_categorical_cross_entropy() {

        let y_pred: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![0.4, 0.4, 0.2]
        ).unwrap();

        let y_true: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![0.0, 1.0, 0.0]
        ).unwrap();

        let result = categorical_cross_entropy(
            &y_pred,
            &y_true 
        ).unwrap();

        let y_pred_2: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![0.1, 0.2, 0.7]
        ).unwrap();

        let y_true_2: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![0.0, 0.0, 1.0]
        ).unwrap();

        let result_2 = categorical_cross_entropy(
            &y_pred_2,
            &y_true_2 
        ).unwrap();


        assert_eq!(result, 0.3054302439580517);
        assert_eq!(result_2, 0.11889164797957748);  
    } 

    #[test]
    fn test_gini_impurity_index() {

        let feature: NDArray<f64> = NDArray::array(
            vec![14, 1],
            vec![
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
            ]
        ).unwrap();

        let gini = gini_impurity(feature);
        assert_eq!(gini, 0.4591836734693877);

    }

}
