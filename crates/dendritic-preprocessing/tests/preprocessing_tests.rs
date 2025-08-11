

#[cfg(test)]
mod preprocessing_tests {

    use ndarray::{arr2, Array2, Axis}; 
    use dendritic_preprocessing::preprocessing::*;

    #[test]
    fn test_one_hot_encoding() {

        let x = arr2(&[
            [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0]
        ]);

        let one_hot = OneHotEncoding::new(&x).unwrap();

        assert_eq!(one_hot.num_classes(), 3); 
        assert_eq!(one_hot.num_samples(), 9); 
        assert_eq!(one_hot.data().dim(), x.dim());
        assert_eq!(one_hot.encoded().dim(), (x.nrows(), 3));
 
    }


    #[test]
    fn test_min_max_scalar() {

    }


    #[test]
    fn test_standard_scalar() {

    }


}
