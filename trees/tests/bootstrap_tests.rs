
#[cfg(test)]
mod bootstrap_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*; 
    use trees::bootstrap::*;
    use metrics::utils::*; 


    #[test]
    fn test_gen_sample() {

        let x_train: NDArray<f64> = NDArray::array(
            vec![8, 2],
            vec![
                69.0, 4.39,
                69.0, 4.21,
                65.0, 4.09,
                72.0, 5.85,
                73.0, 5.68,
                70.0, 5.56,
                73.0, 5.79,
                65.0, 4.27
            ]
        ).unwrap();

        let bs = Bootstrap::new(3, 2, 8, x_train);
        let dataset = bs.sample(8);

        assert_eq!(dataset.shape().values(), vec![8, 2]);
        assert_eq!(dataset.size(), 16);
        assert_eq!(dataset.rank(), 2);

        let sample_one = bs.sample(8);
        let sample_two = bs.sample(8);

        let s1_values = sample_one.values().to_vec(); 
        let s2_values =  sample_two.values().to_vec();

        assert_eq!(s1_values != s2_values, true);
        assert_eq!(s1_values.len(), s2_values.len());

        let s1_col_1 = sample_one.axis(1, 0).unwrap();
        let s2_col_1 = sample_two.axis(1, 0).unwrap();

        assert_eq!(s1_col_1.values() != s2_col_1.values(), true); 
        assert_eq!(s1_col_1.values().len(), s2_col_1.values().len()); 

    }

    #[test]
    fn test_feature_sub_select() {

        let x_path = "data/classification/inputs";
        let y_path = "data/classification/outputs";

        let dataset: NDArray<f64> = NDArray::load(x_path).unwrap(); 
        let target: NDArray<f64> = NDArray::load(y_path).unwrap();

        let mut bs = Bootstrap::new(3, 2, 12, dataset.clone());
        let data = bs.feature_sub_select();
        assert_eq!(data.shape().values(), vec![12, 3]);
        assert_eq!(data.size(), 36);

        let mut bs_2 = Bootstrap::new(3, 3, 12, dataset);
        let data_2 = bs_2.feature_sub_select();
        assert_eq!(data_2.shape().values(), vec![12, 4]);
        assert_eq!(data_2.size(), 48);

        /* validate one of the cols is not the target col */ 
        let target = data_2.axis(1, 3).unwrap();
        let f1 = data_2.axis(1, 0).unwrap(); 
        let f2 = data_2.axis(1, 1).unwrap(); 
        let f3 = data_2.axis(1, 2).unwrap(); 

        assert_ne!(f1.values(), target.values()); 
        assert_ne!(f2.values(), target.values()); 
        assert_ne!(f3.values(), target.values()); 
    }


    #[test]
    fn test_generate() {

        let x_train: NDArray<f64> = NDArray::array(
            vec![8, 2],
            vec![
                69.0, 4.39,
                69.0, 4.21,
                65.0, 4.09,
                72.0, 5.85,
                73.0, 5.68,
                70.0, 5.56,
                73.0, 5.79,
                65.0, 4.27
            ]
        ).unwrap();

        let mut bs = Bootstrap::new(3, 2, 8, x_train);
        bs.generate();

        assert_eq!(bs.datasets().len(), 3);
        assert_eq!(bs.n_bootstraps(), 3); 
        assert_eq!(bs.num_features(), 2); 
        assert_eq!(bs.sample_size(), 8); 

        for item in bs.datasets() {
            assert_eq!(item.shape().values(), vec![8, 2]);
            assert_eq!(item.size(), 16);
        }
        
    }



}
