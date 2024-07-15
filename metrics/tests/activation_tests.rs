
#[cfg(test)]
mod activation_tests {

    use ndarray::ndarray::NDArray;
    use metrics::activations::*;

    #[test]
    fn test_softmax() {

        let input: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![0.25, 1.23, -0.8]
        ).unwrap();

        let output: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![
                0.24910360124886244,  
                0.663725645234628, 
                0.08717075351650967
            ]
        ).unwrap();

        let result = softmax(input);
        assert_eq!(result.values(), output.values());
        
        let input_2: NDArray<f64> = NDArray::array(
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0]
        ).unwrap();

        let output_2: NDArray<f64> = NDArray::array(
            vec![4, 1],
            vec![0.25, 0.25, 0.25, 0.25]
        ).unwrap();

        let result_2 = softmax(input_2);
        assert_eq!(result_2.values(), output_2.values());



    }

    #[test]
    fn test_softmax_prime() {

        let input: NDArray<f64> = NDArray::array(
            vec![3, 1],
            vec![1.0, 1.0, 1.0]
        ).unwrap();

        let result = softmax_prime(input);
        println!("{:?}", result.values()); 

    }

}
