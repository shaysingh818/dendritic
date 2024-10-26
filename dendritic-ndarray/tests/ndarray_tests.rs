use dendritic_ndarray;

#[cfg(test)]
mod ndarray_tests {

    use crate::ndarray::ndarray::NDArray;
    use dendritic_ndarray::ops::*;

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


    #[test]
    fn test_axis() {

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


    #[test]
    fn test_value_indices() {

        let x: NDArray<f64> = NDArray::array(
            vec![6, 1],
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        ).unwrap();

        let idxs = x.value_indices(1.0);
        assert_eq!(idxs, vec![0, 3]);

        let idxs_two = x.value_indices(2.0);
        assert_eq!(idxs_two, vec![1, 4]);

        let idxs_three = x.value_indices(3.0);
        assert_eq!(idxs_three, vec![2, 5]);
        
        let y: NDArray<f64> = NDArray::array(
            vec![6, 1],
            vec![1.0, 1.0, 2.0, 3.0, 4.0, 1.0]
        ).unwrap();

        let y_idxs = y.value_indices(1.0);
        assert_eq!(y_idxs, vec![0, 1, 5]);

        let null_vals = y.value_indices(20.0);
        let len_vec = null_vals.len();
        assert_eq!(len_vec, 0); 

    }

    #[test]
    fn test_indice_query() {

        let x: NDArray<f64> = NDArray::array(
            vec![1, 6],
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        ).unwrap();

        let x_idxs = x.indice_query(vec![1,2,3]).unwrap();
        assert_eq!(*x_idxs.values(), vec![2.0, 3.0, 1.0]);

        let x_idxs_2 = x.indice_query(vec![0,1,2]).unwrap();
        assert_eq!(*x_idxs_2.values(), vec![1.0, 2.0, 3.0]);

        let x_idxs_3 = x.indice_query(vec![3,4,5]).unwrap();
        assert_eq!(*x_idxs_3.values(), vec![1.0, 2.0, 3.0]);

        let x_idxs_out_of_bounds = x.indice_query(vec![1,2,3,4,5,6,7,8,9]);
        assert_eq!(
            x_idxs_out_of_bounds.unwrap_err(),
            "Indices length is greater than array size"
        );

        let x_indice_out_of_bounds = x.indice_query(vec![10, 11, 12]);
        assert_eq!(
            x_indice_out_of_bounds.unwrap_err(),
            "Specified index greater than array size"
        );

    }


    #[test]
    fn test_split() {

        let x_path = "data/split_unit_testing/10_3"; 
        let x: NDArray<f64> = NDArray::load(x_path).unwrap();
        let (x_train, x_test) = x.split(0, 0.80).unwrap();

        assert_eq!(
            x_train.shape().values(),
            vec![8, 3]
        );

        assert_eq!(
            x_test.shape().values(),
            vec![2, 3]
        );

        assert_eq!(
            x.split(10, 0.80).unwrap_err(),
            "AXIS greater than current NDArray shape"
        ); 

        let y_path = "data/split_unit_testing/7_3"; 
        let y: NDArray<f64> = NDArray::load(y_path).unwrap();
        let (y_train, y_test) = y.split(0, 0.50).unwrap();

        assert_eq!(
            y_train.shape().values(),
            vec![4, 3]
        );

        assert_eq!(
            y_test.shape().values(),
            vec![3, 3]
        );
    }

    #[test]
    fn test_drop_axis() {

        let x: NDArray<f64> = NDArray::array(
            vec![3, 3],
            vec![
                1.0, 2.0, 3.0,
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
            ]
        ).unwrap();

        let y: NDArray<f64> = NDArray::array(
            vec![3, 5],
            vec![
                1.0, 2.0, 3.0, 7.0, 8.0,
                2.0, 3.0, 4.0, 7.0, 8.0,
                3.0, 4.0, 5.0, 7.0, 8.0
            ]
        ).unwrap();

        let z: NDArray<f64> = NDArray::array(
            vec![2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ).unwrap();

        let expected_rows = vec![
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0]
        ]; 

        let expected_cols = vec![
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0]
        ]; 

        let expected_y_cols = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![7.0, 7.0, 7.0],
            vec![8.0, 8.0, 8.0],
        ]; 

        let expected_y_rows = vec![
            vec![1.0, 2.0, 3.0, 7.0, 8.0],
            vec![3.0, 4.0, 5.0, 7.0, 8.0],
        ]; 

        let x_1 = x.drop_axis(0, 0).unwrap();
        let x_2 = x.drop_axis(1, 0).unwrap();
        let y_1 = y.drop_axis(0, 1).unwrap();
        let y_2 = y.drop_axis(1, 2).unwrap();

        let bad_idx = x.drop_axis(0, 10); 
        let bad_axis = x.drop_axis(2, 10);
        let bad_array = z.drop_axis(0, 0);

        assert_eq!(
            bad_idx.unwrap_err(),
            "Drop Axis: Selected indice too large for axis"
        );

        assert_eq!(
            bad_axis.unwrap_err(),
            "Drop Axis: Selected axis larger than rank"
        );

        assert_eq!(
            bad_array.unwrap_err(),
            "Drop Axis: Only supported for rank 2 values"
        );

        assert_eq!(
            x_1.shape().values(), 
            vec![2, 3]
        ); 

        assert_eq!(
            x_2.shape().values(), 
            vec![3, 2]
        );

        assert_eq!(
            y_1.shape().values(), 
            vec![2, 5]
        ); 

        assert_eq!(
            y_2.shape().values(), 
            vec![3, 4]
        ); 

        let x_rows = x_1.shape().dim(0);
        let y_rows = y_1.shape().dim(0);
        let y_cols = y_2.shape().dim(1);

        for row in 0..x_rows {

            let item = x_1.axis(0, row).unwrap();
            let item2 = x_2.axis(1, row).unwrap(); 

            assert_eq!(
                item.values(),
                &expected_rows[row]
            );

            assert_eq!(
                item2.values(),
                &expected_rows[row]
            ); 
        }

        for col in 0..y_cols {
            let item = y_2.axis(1, col).unwrap();
            assert_eq!(
                item.values(),
                &expected_y_cols[col]
            );
        }

        for row in 0..y_rows {
            let item = y_1.axis(0, row).unwrap();
            assert_eq!(
                item.values(),
                &expected_y_rows[row]
            );
        }

    }

}


