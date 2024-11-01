#[cfg(test)]
mod elastic {

    use dendritic_ndarray::ndarray::NDArray;
    use dendritic_metrics::loss::*;

    #[test]
    fn test_binary_cross_entropy() {

        let y_pred: NDArray<f64> = NDArray::array(
            vec![10, 1],
            vec![
                0.0, 0.0, 1.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0
            ]
        ).unwrap();

        let y_true: NDArray<f64> = NDArray::array(
            vec![10, 1],
            vec![
                0.19, 0.33, 0.47, 0.7, 0.74,
                0.81, 0.86, 0.94, 0.97, 0.99
            ]
        ).unwrap();

        let result = binary_cross_entropy(&y_true, &y_pred).unwrap();
        println!("{:?}", result); // do something with this later 
        


    }

}
