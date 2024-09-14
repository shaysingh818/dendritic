
#[cfg(test)]
mod aggregate_ops {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*; 


    #[test]
    fn test_unique() {

        let x: NDArray<f64> = NDArray::array(
            vec![4, 1],
            vec![1.0, 2.0, 2.0, 1.0]
        ).unwrap();

        let x_vals = x.unique();

        assert_eq!(x_vals.len(), 2); 
        assert_eq!(x_vals, vec![1.0, 2.0]);

        let y: NDArray<f64> = NDArray::array(
            vec![8, 1],
            vec![3.0, 3.0, 1.0, 2.0, 4.0, 5.0, 7.0, 7.0]
        ).unwrap();

        let y_vals = y.unique();

        assert_eq!(y_vals.len(), 6); 
        assert_eq!(y_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
    }


    #[test]
    fn test_sort() {

        let x: NDArray<f64> = NDArray::array(
            vec![10, 1],
            vec![
                10.0, 9.0, 8.0, 11.0, 12.0, 
                1.0, 2.0, 3.0, 4.0, 5.0
            ]
        ).unwrap();


        let sorted_vals = x.sort();
        let expected = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            8.0, 9.0, 10.0, 11.0, 12.0
        ];

        assert_eq!(sorted_vals.len(), 10);
        assert_eq!(sorted_vals, expected);
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
    fn test_avg_ndarray() {

        let x = NDArray::array(vec![3, 3], vec![
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
        ]).unwrap();


        let y = NDArray::array(vec![9, 1], vec![
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
        ]).unwrap();

        let z = NDArray::array(vec![1, 9], vec![
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
        ]).unwrap();

        assert_eq!(x.avg(), 5.0); 
        assert_eq!(y.avg(), 5.0); 
        assert_eq!(z.avg(), 5.0); 

    }


    #[test]
    fn test_length_ndarray() {

        let x = NDArray::array(vec![2, 2], vec![
            2.0, 2.0, 2.0, 2.0
        ]).unwrap();

        let x_length = x.length();
        assert_eq!(x_length, 4.0);

        let x1 = NDArray::array(vec![1, 2], vec![
            5.0, 0.0
        ]).unwrap();
        let x1_length = x1.length();
        assert_eq!(x1_length, 5.0);

        let x2 = NDArray::array(vec![1, 2], vec![
            0.0, -3.0
        ]).unwrap();
        let x2_length = x2.length();
        assert_eq!(x2_length, 3.0);

    }



} 
