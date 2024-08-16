use ndarray;

#[cfg(test)]
mod ndarray_tests {

    use crate::ndarray::ndarray::NDArray;

    #[test]
    fn test_create_ndarray() {

        let x: NDArray<f64> = NDArray::new(vec![3, 4]).unwrap();
        let shape = x.shape();
        let rank = x.rank();
        let expected_shape : Vec<usize> = vec![3, 4];

        /* asserts */ 
        assert_eq!(rank, 2); 
        assert_eq!(shape.values(), expected_shape);
    }

    #[test]
    fn test_create_ndarray_values() {

        let n: NDArray<f64> = NDArray::array(vec![2, 2], vec![0.0,0.0,1.0,1.0]).unwrap();
        let shape = n.shape();
        let rank = n.rank();
        let values = n.values();
        let expected_shape : Vec<usize> = vec![2, 2];
        let expected_size = 4; 
        let expected_vals = vec![0.0,0.0,1.0,1.0];

        /* asserts */ 
        assert_eq!(shape.values(), expected_shape);
        assert_eq!(rank, 2);
        assert_eq!(values, &expected_vals);
        assert_eq!(expected_size, values.len()); 

        /* value mismatch */ 
        let x1: Result<NDArray<f64>, String> = NDArray::array(vec![3, 4], vec![0.0,0.0,0.0,0.0,1.0,2.0]);
        let expected_error = "Values don't match size based on dimensions"; 
        assert_eq!(x1, Err(expected_error.to_string()));  
    }

    #[test]
    fn test_reshape() {

        /* valid reshape */ 
        let mut n: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        let _ = n.reshape(vec![3, 2]); 
        assert_eq!(n.shape().values(), vec![3, 2]);

        /* rank mismatch */
        let mut x: NDArray<f64> = NDArray::array(vec![2, 4], vec![0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0]).unwrap();
        let bad_reshape = x.reshape(vec![2, 2, 2]);
        let rank_error = "New Shape values don't match rank of array"; 
        assert_eq!(bad_reshape, Err(rank_error.to_string()));

        /* size mismatch */ 
        let bad_size = x.reshape(vec![2, 3]); 
        let size_error = "New Shape values don't match size of array";
        assert_eq!(bad_size, Err(size_error.to_string()));  
    }

    #[test]
    fn test_index() {

        /* 2d indexing */ 
        let mut index = 0;
        let n: NDArray<f64> = NDArray::array(vec![2, 3], vec![0.0,0.0,1.0,1.0,2.0,2.0]).unwrap();
        for i in 0..2 {
            for  j in 0..3 {
                assert_eq!(n.index(vec![i, j]).unwrap(), index);
                index += 1; 
            }
        }

        /* 3d indexing */
        index = 0; 
        let m: NDArray<i32> = NDArray::array(vec![2, 2, 2], vec![0,0,1,1,2,2,3,3]).unwrap();
         for i in 0..2 {
            for  j in 0..2 {
                for k in 0..2 {
                    assert_eq!(m.index(vec![i, j, k]).unwrap(), index);
                    index += 1; 
                }
            }
        }

        /* expected indexing error */
        let o: NDArray<i32> = NDArray::array(vec![2, 3], vec![0,0,1,1,2,2]).unwrap();
        let index_bound = o.index(vec![3, 3]);
        let index_error = "Index out of bounds";
        assert_eq!(index_bound, Err(index_error.to_string()));

        /* expected rank error */ 
        let rank_bound = o.index(vec![0, 0, 1]);
        let rank_error = "Indexing doesn't match rank of ndarray"; 
        assert_eq!(rank_bound, Err(rank_error.to_string()));

    }

    #[test]
    fn test_get_indices() {

        /* 2d indexing */ 
        let mut index = 0;
        let n: NDArray<i32> = NDArray::array(vec![2, 3], vec![0,0,1,1,2,2]).unwrap();
        for i in 0..2 {
            for  j in 0..3 {
                assert_eq!(n.indices(index).unwrap(), vec![i, j]);
                index += 1; 
            }
        }

        index = 0; 
        let m: NDArray<i32> = NDArray::array(vec![2, 2, 2], vec![0,0,1,1,2,2,3,3]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_eq!(m.indices(index).unwrap(), vec![i, j, k]);
                    index += 1; 
                }

            }
        }

        let o: NDArray<i32> = NDArray::array(vec![2, 3], vec![0,0,1,1,2,2]).unwrap();
        let size_bound = o.indices(8);
        let size_error = "Index out of bounds";
        assert_eq!(size_bound, Err(size_error.to_string()));
    }

    #[test]
    fn test_set_ndarray() {

        /* set 2d array */
        let mut index = 0; 
        let mut n: NDArray<i32> = NDArray::array(vec![2, 3], vec![0,0,1,1,2,2]).unwrap();
        for i in 0..2 {
            for  j in 0..3 {
                let _ = n.set(vec![i, j], index);
                index += 1; 
            }
        }

        let n_values = n.values();
        let expected_n : Vec<i32> = vec![0,1,2,3,4,5];
        assert_eq!(n_values, &expected_n);

        /* set 3d array */
        let mut index = 0; 
        let mut n: NDArray<i32> = NDArray::array(vec![2, 2, 2], vec![0,0,1,1,2,2,3,3]).unwrap();
        for i in 0..2 {
            for j in 0..2  {
                for k in 0..2 {
                    let _ = n.set(vec![i, j, k], index);
                    index += 1; 
                }
            }
        }
    }


    #[test]
    fn test_rows_cols_ndarray() {

        /* set 2d array */
        let x: NDArray<f64> = NDArray::array(vec![4, 2], vec![0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0]).unwrap();
        let y: NDArray<f64> = NDArray::array(vec![2, 3], vec![1.0,1.0,1.0,2.0,2.0,2.0]).unwrap();

        /* get rows of x */ 
        let y_row_0: Vec<f64> = y.rows(0).unwrap(); 
        let y_row_1: Vec<f64> = y.rows(1).unwrap();
        let y_col_0: Vec<f64> = y.cols(0).unwrap(); 
        let y_col_1: Vec<f64> = y.cols(1).unwrap();

        let x_row_0: Vec<f64> = x.rows(0).unwrap(); 
        let x_row_1: Vec<f64> = x.rows(1).unwrap();
        let x_col_0: Vec<f64> = x.cols(0).unwrap(); 
        let x_col_1: Vec<f64> = x.cols(1).unwrap();

        /* validate */ 
        assert_eq!(y_row_0, vec![1.0,1.0,1.0]);
        assert_eq!(y_row_1, vec![2.0,2.0,2.0]); 
        assert_eq!(y_col_0, vec![1.0,2.0]);
        assert_eq!(y_col_1, vec![1.0,2.0]);

        assert_eq!(x_col_0, vec![0.0,0.0,1.0,0.0]);
        assert_eq!(x_col_1, vec![0.0,1.0,1.0,0.0]);
        assert_eq!(x_row_0, vec![0.0,0.0]);
        assert_eq!(x_row_1, vec![0.0,1.0]);

    }

    #[test]
    fn test_batch_ndarray() {

        let x: NDArray<f64> = NDArray::array(
            vec![10, 3], 
            vec![
                1.0, 2.0, 3.0,
                2.0, 3.0, 4.0, 
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0, 
                5.0, 6.0, 7.0,
                6.0, 7.0, 8.0,
                7.0, 8.0, 9.0, 
                8.0, 9.0, 10.0,
                9.0, 10.0, 11.0, 
                10.0, 11.0, 12.0
            ]
        ).unwrap();

        let batch_size: usize = 2;
        let x_batch = x.batch(batch_size).unwrap();

        let expected_batches: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 5.0, 6.0, 7.0],
            vec![5.0, 6.0, 7.0, 6.0, 7.0, 8.0],
            vec![6.0, 7.0, 8.0, 7.0, 8.0, 9.0],
            vec![7.0, 8.0, 9.0, 8.0, 9.0, 10.0],
            vec![8.0, 9.0, 10.0, 9.0, 10.0, 11.0],
            vec![9.0, 10.0, 11.0, 10.0, 11.0, 12.0]
        ];

        let mut index = 0; 
        for item in x_batch {
            assert_eq!(item.values(), &expected_batches[index]);
            index += 1; 
        }

    }

}


/*
#[cfg(test)]
mod ops {

    use crate::ndarray::ndarray::NDArray;
    use crate::ndarray::ops::*;

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
    fn test_transpose_ndarray() {

        /* set 2d array */
        let x = NDArray::array(vec![4, 3], vec![1.0,2.0,3.0,1.0,2.0,3.0,0.0,0.0,2.0,0.0,0.0,0.0]).unwrap();
        let x_transpose = x.transpose().unwrap();
        let expected_shape : Vec<usize> = vec![3, 4];
        let expected_vals = vec![1.0,1.0,0.0,0.0,2.0,2.0,0.0,0.0,3.0,3.0,2.0,0.0];

        assert_eq!(x_transpose.shape().values(), expected_shape);
        assert_eq!(x_transpose.rank(), 2);
        assert_eq!(x_transpose.size(), 12);
        assert_eq!(x_transpose.values(), &expected_vals);

        /* failure case */ 
        let y = NDArray::array(vec![2, 2, 2], vec![1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0]).unwrap();
        let expected_shape_two : Vec<usize> = vec![2, 2, 2];
        let expected_vals_two = vec![1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0];

        assert_eq!(y.rank(), 3); 
        assert_eq!(y.values(), &expected_vals_two);
        assert_eq!(y.values().len(), 8);
        assert_eq!(y.shape().values(), expected_shape_two);

        let rank_mismatch = y.transpose();
        match rank_mismatch {
            Ok(_) => println!("Should fail due to rank mismatch"),
            Err(err) => {
                assert_eq!(err, "Transpose must contain on rank 2 values"); 
            }
        }

    }

    #[test]
    fn test_permute_ndarray() {

        /* set 2d array */
        let x = NDArray::array(vec![4, 3], vec![1.0,2.0,3.0,1.0,2.0,3.0,0.0,0.0,2.0,0.0,0.0,0.0]).unwrap();
        let expected_shape : Vec<usize> = vec![3, 4];
        let expected_vals = vec![1.0,1.0,0.0,0.0,2.0,2.0,0.0,0.0,3.0,3.0,2.0,0.0];
        let result = x.permute(vec![1, 0]).unwrap();

        assert_eq!(result.shape().values(), expected_shape);
        assert_eq!(result.rank(), 2);
        assert_eq!(result.size(), 12);
        assert_eq!(result.values(), &expected_vals);

        /* rank mismatch case */ 
        let y = NDArray::array(vec![4, 3], vec![1.0,2.0,3.0,1.0,2.0,3.0,0.0,0.0,2.0,0.0,0.0,0.0]).unwrap(); 
        let rank_mismatch = y.permute(vec![1]);
        match rank_mismatch {
            Ok(_) => println!("Should error out"),
            Err(err) => {
                assert_eq!(err, "Indice order must be same length as rank"); 
            }
        }

    }



    #[test]
    fn test_sum_axis() {

        /* set 2d array */
        let x: NDArray<f64> = NDArray::load("data/linear_modeling_data/inputs").unwrap();
        let y: NDArray<f64> = NDArray::load("data/linear_modeling_data/outputs").unwrap();
        let w: NDArray<f64> = NDArray::load("data/linear_modeling_data/weights").unwrap();
        let b: NDArray<f64> = NDArray::load("data/linear_modeling_data/bias").unwrap();

        let x_sum = x.sum_axis(1).unwrap(); 
        let expected_x_vals = vec![11.0, 20.0, 29.0];
        let expected_shape: Vec<usize> = vec![1, 3];

        assert_eq!(x_sum.values(), &expected_x_vals);
        assert_eq!(x_sum.shape().values(), expected_shape);
        assert_eq!(x_sum.rank(), 2); 

        let y_sum = y.sum_axis(1).unwrap();
        let expected_y_shape: Vec<usize> = vec![1, 1];
        let expected_y_vals = vec![70.0];

        assert_eq!(y_sum.shape().values(), expected_y_shape);
        assert_eq!(y_sum.values(), &expected_y_vals);
        assert_eq!(y_sum.rank(), 2); 


        // test linear operation 
        let dot_op = x.dot(w).unwrap();
        let scale_op = dot_op.scale_add(b).unwrap();
        let error = y.subtract(scale_op).unwrap(); 


        let db = error.sum_axis(1).unwrap();
        let expected_db_shape: Vec<usize> = vec![1, 1];
        let expected_db_vals = vec![65.0];

        assert_eq!(db.shape().values(), expected_db_shape);
        assert_eq!(db.values(), &expected_db_vals);
        assert_eq!(db.rank(), 2); 


        // zero axis sum
        let y_zero_axis_sum = y.sum_axis(0).unwrap();
        let expected_yz_shape: Vec<usize> = vec![1, 1];
        let expected_yz_vals = vec![70.0];

        assert_eq!(y_zero_axis_sum.shape().values(), expected_yz_shape);
        assert_eq!(y_zero_axis_sum.values(), &expected_yz_vals);
        assert_eq!(y_zero_axis_sum.rank(), 2);  

    }


    #[test]
    fn test_square_ndarray() {

        let x = NDArray::array(vec![3, 3], vec![
            2.0,2.0,2.0,
            2.0,2.0,2.0,
            2.0,2.0,2.0,
        ]).unwrap();

        let x_expected = vec![
            4.0,4.0,4.0,
            4.0,4.0,4.0,
            4.0,4.0,4.0,
        ];

        let x_shape = vec![3, 3];
        let expected_shape = vec![3, 1];
        let expected = vec![4.0, 4.0, 4.0];  
        let w_path = "data/ndarray/weights_reg";
        let w: NDArray<f64> = NDArray::load(w_path).unwrap();
        let w_square = w.square().unwrap();

        assert_eq!(w_square.rank(), 2); 
        assert_eq!(w_square.values(), &expected);
        assert_eq!(w_square.shape().values(), expected_shape);

        let x_square = x.square().unwrap();

        assert_eq!(x_square.rank(), 2); 
        assert_eq!(x_square.shape().values(), x_shape); 
        assert_eq!(x_square.values(), &x_expected); 

    } 


    #[test]
    fn test_sum_ndarray() {

        let x = NDArray::array(vec![3, 3], vec![
            2.0,2.0,2.0,
            2.0,2.0,2.0,
            2.0,2.0,2.0,
        ]).unwrap();

        let w_path = "data/ndarray/weights_reg";
        let w: NDArray<f64> = NDArray::load(w_path).unwrap();

        let w_expected = vec![6.0];
        let x_expected = vec![18.0];
        let expected_shape = vec![1, 1];

        let w_square = w.sum().unwrap();

        assert_eq!(w_square.rank(), 2); 
        assert_eq!(w_square.values(), &w_expected);
        assert_eq!(w_square.shape().values(), expected_shape);

        let x_square = x.sum().unwrap();

        assert_eq!(x_square.rank(), 2); 
        assert_eq!(x_square.shape().values(), expected_shape); 
        assert_eq!(x_square.values(), &x_expected); 

    }




    #[test]
    fn test_abs_ndarray() {

        let x = NDArray::array(vec![3, 3], vec![
            2.0,-2.0,3.0,
            -4.0,5.0,-6.0,
            7.0,-8.0,9.0,
        ]).unwrap();

        let expected_abs = vec![
            2.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0
        ];

        let expected_shape = vec![3, 3];
        let x_abs = x.abs().unwrap();

        assert_eq!(x_abs.rank(), 2); 
        assert_eq!(x_abs.shape().values(), expected_shape); 
        assert_eq!(x_abs.values(), &expected_abs); 

        let w = NDArray::array(vec![3, 1], vec![
            0.10,-0.02, -1.32,
        ]).unwrap();

        let expected_w = vec![
            0.10, 0.02, 1.32,
        ];

        let expected_w_shape = vec![3, 1];
        let w_abs = w.abs().unwrap();

        assert_eq!(w_abs.rank(), 2); 
        assert_eq!(w_abs.shape().values(), expected_w_shape); 
        assert_eq!(w_abs.values(), &expected_w); 

    }

    #[test]
    fn test_signum_ndarray() {

        let x = NDArray::array(vec![3, 3], vec![
            2.0,-2.0,3.0,
            -4.0,0.0,-6.0,
            7.0,-8.0,9.0,
        ]).unwrap();

        let expected_shape = vec![3, 3];
        let x_sig = x.signum().unwrap();

        let expected_x = vec![
            1.0, -1.0, 1.0,
            -1.0, 0.0, -1.0,
            1.0, -1.0, 1.0
        ];

        assert_eq!(x_sig.rank(), 2); 
        assert_eq!(x_sig.shape().values(), expected_shape); 
        assert_eq!(x_sig.values(), &expected_x); 

    }


    #[test]
    fn test_mean_ndarray() {

        let x = NDArray::array(vec![4, 3], vec![
            1.0,2.0,3.0,
            2.0,3.0,4.0,
            3.0,4.0,5.0,
            4.0,5.0,6.0
        ]).unwrap();

        let rows_mean = x.mean(0).unwrap();
        let cols_mean = x.mean(1).unwrap();

        let expected_rows_mean = vec![2.0, 3.0, 4.0, 5.0];
        let expected_cols_mean = vec![2.5, 3.5, 4.5];

        assert_eq!(expected_rows_mean, rows_mean); 
        assert_eq!(expected_cols_mean, cols_mean); 

    }

    #[test]
    fn test_stdev_ndarray() {

        let x = NDArray::array(vec![4, 3], vec![
            1.0,2.0,3.0,
            2.0,3.0,4.0,
            3.0,4.0,5.0,
            4.0,5.0,6.0
        ]).unwrap();

        let x_stdev = x.stdev(1).unwrap();
        let expected = vec![
            1.118033988749895,
            1.118033988749895,
            1.118033988749895
        ];

        assert_eq!(x_stdev, expected); 
    }


    #[test]
    fn test_get_axis() {

        let x = NDArray::array(vec![4, 3], vec![
            1.0,2.0,3.0,
            2.0,3.0,4.0,
            3.0,4.0,5.0,
            4.0,5.0,6.0
        ]).unwrap();

        let expected_x_cols = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0]
        ];

        let expected_x_rows = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![4.0, 5.0, 6.0]
        ];


        for row in 0..x.shape().dim(0) {
            let item = x.axis(0, row).unwrap();
            assert_eq!(item.values(), &expected_x_rows[row]);
        }

        for col in 0..x.shape().dim(1) {
            let item = x.axis(1, col).unwrap();
            assert_eq!(item.values(), &expected_x_cols[col]);
        }  

        let x_4d = NDArray::load("data/ndarray/4d_array").unwrap();

        let x_4d_row = x_4d.axis(0, 0).unwrap();
        let x_4d_row_expected = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
        ];
        assert_eq!(x_4d_row.values(), &x_4d_row_expected);


        let x_4d_col_expected = vec![
            0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0,
            17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0
        ];
        let x_4d_col = x_4d.axis(1, 0).unwrap();
        assert_eq!(x_4d_col.values(), &x_4d_col_expected); 

        let x_4d_3_expected = vec![
            0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0,
            16.0, 17.0, 20.0, 21.0, 24.0, 25.0, 28.0, 29.0
        ];
        let x_4d_3 = x_4d.axis(2, 0).unwrap();
        assert_eq!(x_4d_3.values(), &x_4d_3_expected); 

        let x_4d_4_expected = vec![
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
            16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0
        ];
        let x_4d_4 = x_4d.axis(3, 0).unwrap();
        assert_eq!(x_4d_4.values(), &x_4d_4_expected);  

        let x_5d = NDArray::load("data/ndarray/5d_array").unwrap();
        let x_5d_col = x_5d.axis(1, 0).unwrap();
        let expected_x_5d_col: Vec<f64> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 
            18.0, 19.0, 20.0, 21.0, 22.0,23.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 
            80.0, 81.0, 82.0, 83.0,84.0, 85.0, 86.0, 87.0, 
            88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0
        ];

        assert_eq!(x_5d_col.shape().values(), vec![2, 2, 4, 3]);
        assert_eq!(x_5d_col.values(), &expected_x_5d_col);

    }  


}

*/
