
#[cfg(test)]
mod utils_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::UtfOps;  
    use metrics::utils::*; 
    use metrics::activations::*;

    #[test]
    fn test_apply_axis() {

        let input: NDArray<f64> = NDArray::array(
            vec![3, 3],
            vec![
                1.0, 1.0, 1.0,
                1.0, 2.0, 3.0,
                3.0, 4.0, 5.0
            ]
        ).unwrap();


        let expected_vec = vec![
            0.333, 0.333, 0.333,
            0.09, 0.245, 0.665,
            0.09, 0.245, 0.665
        ];

        let softmax_rows = apply(input, 0, softmax);
        let softmax_vec = softmax_rows.values();
        let rounded: Vec<f64> = softmax_vec.iter().map(
            |x| (x * 1000.0).round() / 1000.0
        ).collect();

        assert_eq!(&rounded, &expected_vec);


        let test: NDArray<f64> = NDArray::array(
            vec![2, 3],
            vec![
                2.0, 1.0, 0.1,
                1.0, 3.0, 0.3,
            ]
        ).unwrap();

        let test_rows = apply(test, 0, softmax);
        //println!("{:?}", test_rows.values()); 
    }


    #[test]
    fn test_entropy() {
        
        let a: NDArray<f64> = NDArray::array(
            vec![9, 1],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        ).unwrap();

        let a_entropy = entropy(a);
        assert_eq!(a_entropy, 1.584962500721156);

        let b: NDArray<f64> = NDArray::array(
            vec![10, 1],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        ).unwrap();

        let b_entropy = entropy(b);
        assert_eq!(b_entropy, 1.0);
    
    }


}
