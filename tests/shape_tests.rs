use steepgrad::ndarray;

#[cfg(test)]
mod shape_tests {

    use crate::ndarray::shape::*;

    #[test]
    fn test_stride_shape() {

        let x: Shape = Shape::new(vec![3, 4]);
        let x_rows = x.stride(0); 
        let x_cols = x.stride(1); 

        assert_eq!(x_rows, 1); 
        assert_eq!(x_cols, 4); 

        let y: Shape = Shape::new(vec![3, 4, 5]);
        let y_rows = y.stride(0); 
        let y_cols = y.stride(1); 
        let y_3d = y.stride(2);

        assert_eq!(y_rows, 1); 
        assert_eq!(y_cols, 5); 
        assert_eq!(y_3d, 20); 
 
    }

    #[test]
    fn test_axis_length_shape() {

        let x: Shape = Shape::new(vec![4, 3]);
        let x_rows = x.axis_length(0); 
        let x_cols = x.axis_length(1);

        assert_eq!(x_rows, 4); 
        assert_eq!(x_cols, 3); 

        let y: Shape = Shape::new(vec![4, 2, 2]);
        let y_rows = y.axis_length(0); 
        let y_cols = y.axis_length(1);
        let y_3 = y.axis_length(2);

        assert_eq!(y_rows, 4); 
        assert_eq!(y_cols, 2);
        assert_eq!(y_3, 4);

    }

}
