use dendritic_ndarray;

#[cfg(test)]
mod shape_tests {

    use crate::dendritic_ndarray::shape::*;

    /*
    #[test]
    fn test_stride_shape() {

        let x: Shape = Shape::new(vec![2,3]);
        let x_strides = x.strides();
        let expected_x_strides = vec![24, 8];
        assert_eq!(x_strides, expected_x_strides);


        let z: Shape = Shape::new(vec![2,3,2,4,3]);
        let strides = z.strides();
        let expected_z_strides = vec![576, 192, 96, 24, 8];
        assert_eq!(strides, expected_z_strides);

 
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

        let z: Shape = Shape::new(vec![2, 2, 2, 2]);
        let z_rows = z.axis_length(0); 
        let z_cols = z.axis_length(1);
        let z_3 = z.axis_length(2);
        let z_4 = z.axis_length(3);

        println!(
            "{:?} {:?} {:?} {:?}",
            z_rows, z_cols, z_3, z_4
        );

    } */ 

}
