

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





}
