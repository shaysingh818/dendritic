

#[cfg(test)]
mod ndarray_ops {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*; 

    #[test]
    fn test_dot_theorem() {

        let x: NDArray<f64> = NDArray::new(vec![3, 4]).unwrap();
        let shape = x.shape();
        let rank = x.rank();
        let expected_shape : Vec<usize> = vec![3, 4];

        /* asserts */ 
        assert_eq!(rank, 2); 
        assert_eq!(shape.values(), expected_shape);
    }

    #[test]
    fn test_argmax() {

        let x: NDArray<f64> = NDArray::array(
            vec![3, 4],
            vec![
                1.0, 1.0, 1.0, 1.0,
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 1.0, 0.0
            ]
        ).unwrap(); 

        let result = x.argmax(0);
        let expected = vec![0.0, 3.0, 1.0];
        assert_eq!(result.values(), &expected); 

    }



}
