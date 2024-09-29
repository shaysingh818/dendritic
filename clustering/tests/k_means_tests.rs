
#[cfg(test)]
mod kmeans_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use knn::distance::*;
    use clustering::k_means::*;

    #[test]
    fn test_kmeans() {

        let k_value = 2; 
        let data_path = "data/k_means_unit/data";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();
        assert_eq!(data.shape().values(), vec![12, 2]); 

        let clf = KMeans::fit(&data, k_value, euclidean).unwrap();

        assert_eq!(clf.data.shape().values(), vec![12, 2]);
        assert_eq!(clf.k, 2);


        let clf_bad = KMeans::fit(&data, 100, euclidean);
        assert_eq!(
            clf_bad.unwrap_err(),
            "Not enough rows in sample data"
        );
    }


    #[test]
    fn test_assign_clusters() {

        let k_value = 2; 
        let data_path = "data/k_means_unit/data";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();
        assert_eq!(data.shape().values(), vec![12, 2]); 

        let clf = KMeans::fit(&data, k_value, euclidean).unwrap();

        assert_eq!(clf.data.shape().values(), vec![12, 2]);
        assert_eq!(clf.k, 2);

        clf.assign_clusters();


    }

}
