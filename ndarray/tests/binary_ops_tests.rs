
#[cfg(test)]
mod binary_ops {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;


    #[test]
    fn test_add_ndarray() {

        /* set 2d array */
        let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        let y: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();

        let result : NDArray<f64> = x.add(y).unwrap(); 
        let expected_vals = vec![0.0,0.0,2.0,2.0,4.0,4.0];
        let expected_shape = vec![2, 3]; 

        assert_eq!(result.rank(), 2); 
        assert_eq!(result.values(), &expected_vals);
        assert_eq!(result.values().len(), 6);
        assert_eq!(result.shape().values(), expected_shape);
        
        /* failure case */
        let z: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        let o: NDArray<f64> = NDArray::array(vec![2, 2], vec![0.0,0.0,1.0,1.0]).unwrap();
        let result: Result<NDArray<f64>, String> = o.add(z); // catch the error
        match result {
            Ok(_) => println!("This should fail"), 
            Err(err) => {
                assert_eq!(err, "Add: Size mismatch for arrays"); 
            }
        }

        let a: NDArray<f64> = NDArray::array(vec![2, 2, 2], vec![0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0]).unwrap();
        let b: NDArray<f64> = NDArray::array(vec![2, 4], vec![0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0]).unwrap();
        let rank_mismatch: Result<NDArray<f64>, String> = a.add(b); 
        match rank_mismatch {
            Ok(_) => println!("Fail due to rank mismatch"),
            Err(err) => {
                assert_eq!(err, "Add: Rank Mismatch"); 
            }
        }

    }


    #[test]
    fn test_subtract_ndarray() {

        /* set 2d array */
        let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        let y: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();

        let result : NDArray<f64> = x.subtract(y).unwrap(); 
        let expected_vals = vec![0.0,0.0,0.0,0.0,0.0,0.0];
        let expected_shape = vec![2, 3]; 

        assert_eq!(result.rank(), 2); 
        assert_eq!(result.values(), &expected_vals);
        assert_eq!(result.values().len(), 6);
        assert_eq!(result.shape().values(), expected_shape);
        
        /* failure case */
        let z: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        let o: NDArray<f64> = NDArray::array(vec![2, 2], vec![0.0,0.0,1.0,1.0]).unwrap();
        let result: Result<NDArray<f64>, String> = o.subtract(z); // catch the error
        match result {
            Ok(_) => println!("This should fail"), 
            Err(err) => {
                assert_eq!(err, "Subtract: Size mismatch for arrays"); 
            }
        }

        let a: NDArray<f64> = NDArray::array(vec![2, 2, 2], vec![0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0]).unwrap();
        let b: NDArray<f64> = NDArray::array(vec![2, 4], vec![0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0]).unwrap();
        let rank_mismatch: Result<NDArray<f64>, String> = a.subtract(b); 
        match rank_mismatch {
            Ok(_) => println!("Fail due to rank mismatch"),
            Err(err) => {
                assert_eq!(err, "Subtract: Rank Mismatch"); 
            }
        }
        
    }

    
    #[test]
    fn test_dot_ndarray() {

        /* set 2d array */
        let a: NDArray<f64> = NDArray::array(vec![4, 3], vec![0.0,0.0,1.0,0.0,1.0,2.0,1.0,1.0,3.0,0.0,0.0,4.0]).unwrap();
        let b: NDArray<f64> = NDArray::array(vec![3, 3], vec![1.0,1.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0]).unwrap();

        let result : NDArray<f64> = a.dot(b).unwrap(); 
        let expected_vals = vec![3.0,3.0,3.0,8.0,8.0,8.0,12.0,12.0,12.0,12.0,12.0,12.0];
        let expected_shape = vec![4, 3];
        
        assert_eq!(result.values(), &expected_vals);
        assert_eq!(result.shape().values(), expected_shape);
        assert_eq!(result.values().len(), 12); 
        assert_eq!(result.rank(), 2);
        
        /* set 2d array */
        let x: NDArray<f64> = NDArray::array(vec![4, 2], vec![0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0]).unwrap();
        let y: NDArray<f64> = NDArray::array(vec![2, 3], vec![1.0,1.0,1.0,2.0,2.0,2.0]).unwrap();

        let result : NDArray<f64> = x.dot(y).unwrap(); 
        let expected_vals = vec![0.0,0.0,0.0,2.0,2.0,2.0,3.0,3.0,3.0, 0.0, 0.0, 0.0];
        let expected_shape = vec![4, 3];
        

        assert_eq!(result.values(), &expected_vals);
        assert_eq!(result.shape().values(), expected_shape);
        assert_eq!(result.values().len(), 12); 
        assert_eq!(result.rank(), 2);

            
        /* failure case */
        let z: NDArray<f64> = NDArray::array(vec![2,2,2], vec![0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0]).unwrap();
        let o: NDArray<f64> = NDArray::array(vec![2, 2], vec![0.0,0.0,1.0,1.0]).unwrap();
        let result: Result<NDArray<f64>, String> = o.dot(z); // catch the error
        match result {
            Ok(_) => println!("This should fail"), 
            Err(err) => {
                assert_eq!(err, "Dot: Rank Mismatch"); 
            }
        }


        let m: NDArray<f64> = NDArray::array(vec![2, 4], vec![0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0]).unwrap();
        let p: NDArray<f64> = NDArray::array(vec![2, 3], vec![1.0,1.0,1.0,2.0,2.0,2.0]).unwrap();
        let rank_mismatch: Result<NDArray<f64>, String> = m.dot(p); 
        match rank_mismatch {
            Ok(_) => println!("Fail due to rank mismatch"),
            Err(err) => {
                assert_eq!(err, "Dot: Rows must equal columns"); 
            }
        }

        let a_path = "data/ndarray/X_T";
        let b_path = "data/ndarray/Y_P";

        let a = NDArray::load(a_path).unwrap();
        let b = NDArray::load(b_path).unwrap();

        let dot_result = a.dot(b).unwrap();
        let expected = vec![55.0, 110.0, 165.0, 55.0, 110.0, 165.0, 55.0, 110.0, 165.0];

        assert_eq!(dot_result.shape().values(), vec![3,3]);
        assert_eq!(dot_result.values(), &expected);
        
    }


    #[test]
    fn test_scale_add_ndarray() {

        /* set 2d array */
        let x: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        let y: NDArray<f64> = NDArray::array(vec![1, 3], vec![1.0,1.0,1.0]).unwrap();

        let result : NDArray<f64> = x.scale_add(y).unwrap(); 
        let expected_vals = vec![1.0,1.0,2.0,2.0,3.0,3.0];
        let expected_shape = vec![2, 3]; 

        assert_eq!(result.rank(), 2); 
        assert_eq!(result.values(), &expected_vals);
        assert_eq!(result.values().len(), 6);
        assert_eq!(result.shape().values(), expected_shape);
        
        /* set 2d array */
        let a = NDArray::array(vec![4, 3], vec![0.0,0.0,0.0,2.0,2.0,2.0,2.0,2.0,2.0,4.0,4.0,4.0]).unwrap();
        let b = NDArray::array(vec![1, 3], vec![1.0,1.0,1.0]).unwrap();

        let result_two : NDArray<f64> = a.scale_add(b).unwrap(); 
        let expected_vals_two = vec![1.0,1.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0,5.0,5.0,5.0];
        let expected_shape_two = vec![4, 3]; 

        assert_eq!(result_two.rank(), 2); 
        assert_eq!(result_two.values(), &expected_vals_two);
        assert_eq!(result_two.values().len(), 12);
        assert_eq!(result_two.shape().values(), expected_shape_two);

        /* failure case */
        let o = NDArray::array(vec![4, 3], vec![0.0,0.0,0.0,2.0,2.0,2.0,2.0,2.0,2.0,4.0,4.0,4.0]).unwrap();
        let n = NDArray::array(vec![2, 2], vec![1.0,1.0,1.0,1.0]).unwrap();
        let result_bad: Result<NDArray<f64>, String> = o.scale_add(n);
        match result_bad {
            Ok(_) => println!("Fail due to dimension mismatch"),
            Err(err) => {
                assert_eq!(err, "Scale add must have a vector dimension (1, N)"); 
            }
        }

    }


    #[test]
    fn test_save_load_ndarray() {

        /* for logistic data */
        let inputs_logistic = vec![
            1.0, 2.0, 3.0, -1.0, 2.0, 1.0, 2.0, -2.0, 0.0, -1.0, 
            -1.0, 2.0, 1.0, 1.0, 0.0, -2.0, 2.0, 1.0, 1.0, -1.0,
            0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0
        ];

        /* set 2d array */
        let log_x: NDArray<f64> = NDArray::array(vec![10, 3], inputs_logistic.clone()).unwrap();
        let _ = log_x.save("data/ndarray/saved");

        /* load from saved ndarray */
        let loaded: NDArray<f64> = NDArray::load("data/ndarray/saved").unwrap();
        let shape = loaded.shape();
        let rank = loaded.rank();
        let values = loaded.values();

        /* expected attributes */
        assert_eq!(shape.values(), vec![10, 3]);
        assert_eq!(values.len(), 30);
        assert_eq!(values, &inputs_logistic);
        assert_eq!(rank, 2); 

    } 

}
