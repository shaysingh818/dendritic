

#[cfg(test)]
mod preprocessing_tests {

    use dendritic_ndarray::ndarray::NDArray;
    use dendritic_preprocessing::standard_scalar::*;

    #[test]
    fn test_min_max_scalar() {

        let x = NDArray::array(vec![4, 3], vec![
            1.0,2.0,31.0,
            2.0,3.0,41.0,
            3.0,4.0,51.0,
            4.0,5.0,61.0
        ]).unwrap();


        let min_max = min_max_scalar(x.clone()).unwrap();

        let min = min_max.values().iter().min_by(
            |a, b| a.total_cmp(b)
        ).unwrap();

        let max = min_max.values().iter().max_by(
            |a, b| a.total_cmp(b)
        ).unwrap();

        assert_eq!(*min, 0.0); 
        assert_eq!(*max, 1.0);
        assert_eq!(min_max.shape().values(), x.shape().values());

        let rounded: Vec<f64> = min_max.values().iter().map(
            |x| (x * 1000.0).round() / 1000.0
        ).collect();


        let expected_vals = vec![
            0.0,0.0,0.0,
            0.333,0.333,0.333,
            0.667,0.667,0.667,
            1.0,1.0,1.0
        ];

        assert_eq!(expected_vals, rounded); 
 
    }



}
