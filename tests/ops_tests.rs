use steepgrad::ndarray;


#[cfg(test)]
mod ndarray_ops {

    use crate::ndarray::ndarray::NDArray;

    #[test]
    fn test_dot_theorem() {

        let mut x: NDArray<f64> = NDArray::new(vec![3, 4]).unwrap();
        let shape = x.shape();
        let rank = x.rank();
        let expected_shape : Vec<usize> = vec![3, 4];

        /* asserts */ 
        assert_eq!(rank, 2); 
        assert_eq!(shape, &expected_shape);

    }

}