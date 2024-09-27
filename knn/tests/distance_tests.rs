
#[cfg(test)]
mod distance_tests {

    use metrics::loss::*; 
    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use knn::distance::*;

    #[test]
    fn test_euclidean_distance() {

        let p1: NDArray<f64> = NDArray::array(
            vec![1, 2],
            vec![20.0, 35.0]
        ).unwrap();

        let p2: NDArray<f64> = NDArray::array(
            vec![1, 2],
            vec![40.0, 20.0]
        ).unwrap();

        let p3: NDArray<f64> = NDArray::array(
            vec![1, 3],
            vec![40.0, 20.0, 60.0]
        ).unwrap();

        let distance = euclidean(&p1, &p2).unwrap();
        assert_eq!(distance, 25.0); 

        let bad_distance = euclidean(&p1, &p3);
        assert_eq!(
            bad_distance.unwrap_err(),
            "Supplied points must be of same shape"
        );

    }


    #[test]
    fn test_manhattan_distance() {

        let p1: NDArray<f64> = NDArray::array(
            vec![1, 3],
            vec![1.0, 2.0, 3.0]
        ).unwrap();

        let p2: NDArray<f64> = NDArray::array(
            vec![1, 3],
            vec![4.0, 5.0, 6.0]
        ).unwrap();

        let p3: NDArray<f64> = NDArray::array(
            vec![1, 2],
            vec![40.0, 20.0]
        ).unwrap();

        let distance = manhattan(&p1, &p2).unwrap();
        assert_eq!(distance, 9.0); 

        let bad_distance = manhattan(&p1, &p3);
        assert_eq!(
            bad_distance.unwrap_err(),
            "Supplied points must be of same shape"
        ); 

    }

}
