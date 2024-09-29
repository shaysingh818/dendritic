
#[cfg(test)]
mod unary_ops {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;


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
    fn test_axis_indices() {

        let x: NDArray<f64> = NDArray::array(
            vec![10, 4],
            vec![
                1.0, 1.0, 1.0, 1.0,
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 2.0, 3.0, 4.0,
                2.0, 2.0, 1.0, 0.0,
                2.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 3.0, 4.0,
                2.0, 2.0, 1.0, 10.0,
                3.0, 2.0, 1.0, 10.0,
            ]
        ).unwrap();


        let indices: Vec<usize> = vec![0, 9, 8, 7, 6];
        let rows = x.axis_indices(0, indices).unwrap();

        assert_eq!(rows.shape().values(), vec![5, 4]);
        assert_eq!(rows.size(), 20);

        let col_0 = rows.axis(1, 0).unwrap();
        let col_1 = rows.axis(1, 1).unwrap();
        let col_2 = rows.axis(1, 2).unwrap();
        let col_3 = rows.axis(1, 3).unwrap();

        assert_eq!(col_0.values(), &vec![1.0, 3.0, 2.0, 2.0, 2.0]);
        assert_eq!(col_1.values(), &vec![1.0, 2.0, 2.0, 2.0, 1.0]);
        assert_eq!(col_2.values(), &vec![1.0, 1.0, 1.0, 3.0, 1.0]);
        assert_eq!(col_3.values(), &vec![1.0, 10.0, 10.0, 4.0, 1.0]);

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


    #[test]
    fn test_argmin() {

        let x: NDArray<f64> = NDArray::array(
            vec![3, 3],
            vec![
                1.0, 2.0, 0.0,
                0.0, 2.0, 1.0,
                1.0, 0.0, 2.0,
            ]
        ).unwrap();

        x.argmin(0);


    }

    #[test]
    fn test_select_axis() {

        let x: NDArray<f64> = NDArray::array(
            vec![4, 4],
            vec![
                1.0, 1.0, 1.0, 0.0,
                2.0, 2.0, 2.0, 1.0,
                3.0, 3.0, 3.0, 0.0,
                4.0, 4.0, 4.0, 1.0,
            ]
        ).unwrap();

        let expected_x_vals = vec![
            1.0, 1.0, 0.0, 2.0, 2.0, 1.0,
            3.0, 3.0, 0.0, 4.0, 4.0, 1.0
        ]; 

        let expected_x_cols = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 0.0, 1.0]
        ];

        let expected_y_cols = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0]
        ];

        let expected_z_rows = vec![
            vec![2.0, 2.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 0.0],
            vec![4.0, 4.0, 4.0, 1.0]
        ];

        let x_cols = x.select_axis(1, vec![0, 1, 3]).unwrap();
        let y_cols = x.select_axis(1, vec![3, 1, 0]).unwrap(); 
        let z_rows = x.select_axis(0, vec![1, 0, 3]).unwrap();

        assert_eq!(x_cols.values(), &expected_x_vals);
        assert_eq!(x_cols.shape().values(), vec![4, 3]);
        assert_eq!(y_cols.shape().values(), vec![4, 3]); 
        assert_eq!(z_rows.shape().values(), vec![3, 4]); 

        let mut counter = 0; 
        for col in 0..3 {
            let x_col = x_cols.axis(1, col).unwrap(); 
            let y_col = y_cols.axis(1, col).unwrap();
            let z_row = z_rows.axis(0, col).unwrap();

            assert_eq!(x_col.values(), &expected_x_cols[col]);
            assert_eq!(y_col.values(), &expected_y_cols[col]); 
            assert_eq!(z_row.values(), &expected_z_rows[col]); 

            counter += 1; 
        }


    }


}
