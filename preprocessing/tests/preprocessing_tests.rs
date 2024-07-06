

#[cfg(test)]
mod preprocessing_tests {

    use ndarray::ndarray::NDArray;
    use preprocessing::standard_scalar::*;

    #[test]
    fn test_standard_scalar() {

        let x = NDArray::array(vec![4, 3], vec![
            1.0,2.0,31.0,
            2.0,3.0,41.0,
            3.0,4.0,51.0,
            4.0,5.0,61.0
        ]).unwrap();

        let std_scalar = standard_scalar(x.clone()).unwrap();
        println!("{:?}", std_scalar.values()); 

        let min_max = min_max_scalar(x).unwrap();
        println!("{:?}", min_max.values()); 
        
        
    }



}
