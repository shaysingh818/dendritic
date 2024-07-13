
#[cfg(test)]
mod utils_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
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
        println!("{:?}", test_rows.values()); 


    }


}
