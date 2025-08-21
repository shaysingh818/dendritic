

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

        let decoded = one_hot.decode(&encoded);
        assert_eq!(decoded, x); 
 
    }


    #[test]
    fn test_standard_scalar() {

        let x = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]);

        let mut scalar = StandardScalar::new(&x).unwrap();

        assert_eq!(scalar.data(), x); 
        assert_eq!(scalar.data().dim(), x.dim());
        assert_eq!(scalar.mean().len(), 2); 
        assert_eq!(scalar.stdev().len(), 2);

        let encoded_data = scalar.encode(); 

        assert_eq!(
            encoded_data.mapv(|x| (x * 10000.0).round() / 10000.0),
            arr2(&[
                [-1.4142, -1.4142],
                [-0.7071, -0.7071],
                [ 0.0000,  0.0000],
                [ 0.7071,  0.7071],
                [ 1.4142,  1.4142]
            ])
        );

        assert_eq!(scalar.decode(&encoded_data), x);

    }


    #[test]
    fn test_min_max_scalar() {

        let x = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]);

        let expected = arr2(&[
            [0.0, 0.0],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75],
            [1.0, 1.0]
        ]); 

        let mut min_max = MinMaxScalar::new(&x).unwrap();

        assert_eq!(min_max.data().dim(), x.dim()); 
        assert_eq!(min_max.data(), &x);
        assert_eq!(min_max.min_range().len(), x.ncols());
        assert_eq!(min_max.max_range().len(), x.ncols());

        let encoded = min_max.encode();

        assert_eq!(min_max.min_range(), &vec![1.0, 2.0]); 
        assert_eq!(min_max.max_range(), &vec![5.0, 10.0]);
        assert_eq!(&encoded, expected);

        let decoded = min_max.decode(&encoded);
        assert_eq!(decoded, x);

    }


}
