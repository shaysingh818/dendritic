
#[cfg(test)]
mod encoding_tests {

    use ndarray::ndarray::NDArray;
    use preprocessing::encoding::{OneHotEncoding};

    #[test]
    fn test_one_hot_encoder() {

        let x = NDArray::array(vec![10, 1], vec![
            1.0,2.0,0.0,2.0,0.0,
	    0.0,1.0,0.0,2.0,2.0
        ]).unwrap();

        let mut encoder = OneHotEncoding::new(x).unwrap();
        assert_eq!(encoder.max_value(), 3.0); 
        assert_eq!(encoder.num_samples(), 10.0); 

        let encoded_vals = encoder.transform();

        let expected = vec![
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ];

        assert_eq!(encoded_vals.shape().values(), vec![10, 3]);
        assert_eq!(encoded_vals.values(), &expected); 
        
    }



}
