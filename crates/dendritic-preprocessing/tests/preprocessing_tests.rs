

#[cfg(test)]
mod preprocessing_tests {

    use ndarray::{arr2, Array2, Axis}; 
    use dendritic_preprocessing::preprocessing::*;

    #[test]
    fn test_one_hot_encoding() {

        let x = arr2(&[
            [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0]
        ]);

        let encoded = arr2(&[
            [1.0,0.0,0.0],
            [1.0,0.0,0.0],
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,1.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0],
            [0.0,0.0,1.0],
            [0.0,0.0,1.0]
        ]);

        let mut one_hot = OneHotEncoding::new(&x).unwrap();
        let bad_on_hot = OneHotEncoding::new(
            &arr2(&[
                 [0.0, 0.0],
                 [0.0, 0.0]
            ])
        );

        assert_eq!(one_hot.num_classes(), 3); 
        assert_eq!(one_hot.num_samples(), 9); 
        assert_eq!(one_hot.data().dim(), x.dim());

        assert_eq!(
            bad_on_hot.unwrap_err().to_string(),
            "Input must be of size (N, 1)"
        );

        assert_eq!(one_hot.encode(), encoded);

        let decoded = one_hot.decode(encoded);
        assert_eq!(decoded, x); 
 
    }


    #[test]
    fn test_standard_scalar() {

        let x = arr2(&[
            [1.0,2.0,1.0],
            [2.0,2.0,6.0],
            [3.0,3.0,5.0],
            [4.0,1.0,4.0],
            [5.0,1.0,3.0],
            [6.0,1.0,2.0],
            [7.0,4.0,1.0]
        ]);

        let mut standard_scalar = StandardScalar::new(&x).unwrap();

        assert_eq!(standard_scalar.data(), x); 
        assert_eq!(standard_scalar.data().dim(), x.dim());
        assert_eq!(standard_scalar.mean().len(), 3); 
        assert_eq!(standard_scalar.stdev().len(), 3);

        let data = standard_scalar.encode(); 



    }


    #[test]
    fn test_min_max_scalar() {

    }


}
