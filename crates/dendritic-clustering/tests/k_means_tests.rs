
#[cfg(test)]
mod kmeans_tests {

    use dendritic_ndarray::ndarray::NDArray;
    use dendritic_ndarray::ops::*;
    use dendritic_knn::distance::*;
    use dendritic_clustering::k_means::*;
    use dendritic_datasets::iris::*; 

    #[test]
    fn test_kmeans() {

        let k_value = 2; 
        let data_path = "data/k_means_unit/data";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();
        assert_eq!(data.shape().values(), vec![12, 2]); 

        let clf = KMeans::new(&data, k_value, 5, euclidean).unwrap();

        assert_eq!(clf.data.shape().values(), vec![12, 2]);
        assert_eq!(clf.k, 2);

        let expected_centroids = vec![
            vec![185.0, 72.0],
            vec![170.0, 56.0]
        ];

        let mut idx = 0;
        for centroid in clf.centroids() {
            assert_eq!(
                centroid.values(),
                &expected_centroids[idx]
            );
            idx += 1;
        }


        let clf_bad = KMeans::new(&data, 100, 5, euclidean);
        assert_eq!(
            clf_bad.unwrap_err(),
            "Not enough rows in sample data"
        );
    }


    #[test]
    fn test_assign_clusters() {

        let k_value = 3; 
        let data_path = "data/k_means_unit/data2";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();
        assert_eq!(data.shape().values(), vec![8, 2]);

        let mut clf = KMeans::new(&data, k_value, 5, euclidean).unwrap();
        assert_eq!(clf.data.shape().values(), vec![8, 2]);
        assert_eq!(clf.k, 3);

        let mut expected_centroids = vec![
            vec![2.0, 10.0],
            vec![2.0, 5.0],
            vec![8.0, 4.0]
        ];

        let mut idx = 0;
        for centroid in clf.centroids() {
            assert_eq!(
                centroid.values(),
                &expected_centroids[idx]
            );
            idx += 1;
        }

        let idxs: Vec<usize> = vec![0, 3, 6];
        clf.set_centroids(&idxs); 

        expected_centroids = vec![
            vec![2.0, 10.0],
            vec![5.0, 8.0],
            vec![1.0, 2.0]
        ];

        idx = 0;
        for centroid in clf.centroids() {
            assert_eq!(
                centroid.values(),
                &expected_centroids[idx]
            );
            idx += 1;
        }

        let results = clf.assign_clusters();
        assert_eq!(results.shape().values(), vec![8, 1]);
        assert_eq!(
            results.values(),
            &vec![0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
        );

    }


    #[test]
    fn test_calculate_centroids() {

        let k_value = 3; 
        let data_path = "data/k_means_unit/data2";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();
        assert_eq!(data.shape().values(), vec![8, 2]);

        let mut clf = KMeans::new(&data, k_value, 5, euclidean).unwrap();
        assert_eq!(clf.data.shape().values(), vec![8, 2]);
        assert_eq!(clf.k, 3);

        let idxs: Vec<usize> = vec![0, 3, 6];
        clf.set_centroids(&idxs);

        let mut expected_centroids = vec![
            vec![2.0, 10.0],
            vec![5.0, 8.0],
            vec![1.0, 2.0]
        ];

        let mut idx = 0;
        for centroid in clf.centroids() {
            assert_eq!(
                centroid.values(),
                &expected_centroids[idx]
            );
            idx += 1;
        }

        let assigned = clf.assign_clusters();
        clf.calculate_centroids(&assigned);

        expected_centroids = vec![
            vec![2.0, 10.0],
            vec![6.0, 6.0],
            vec![1.5, 3.5]
        ];

        idx = 0;
        for centroid in clf.centroids() {
            assert_eq!(
                centroid.values(),
                &expected_centroids[idx]
            );
            idx += 1;
        }

    }

    #[test]
    fn test_fit_kmeans() {

        let k_value = 3; 
        let data_path = "data/k_means_unit/data2";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();
        assert_eq!(data.shape().values(), vec![8, 2]);

        let mut clf = KMeans::new(&data, k_value, 5, euclidean).unwrap();
        assert_eq!(clf.data.shape().values(), vec![8, 2]);
        assert_eq!(clf.k, 3);

        let idxs: Vec<usize> = vec![0, 3, 6];
        clf.set_centroids(&idxs);

        let expected = vec![0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0];
        let final_centroids = clf.fit();

        assert_eq!(final_centroids.shape().values(), vec![8, 1]); 
        assert_eq!(
            final_centroids.values(),
            &expected
        );

    }

    #[test]
    fn test_iris_clusters() {

        // load data
        let k_value = 3; 
        let data_path = "../dendritic-datasets/data/iris.parquet";
        let (x_train, y_train) = load_iris(data_path).unwrap();

        let mut clf = KMeans::new(
            &x_train, 
            k_value,
            50,
            euclidean
        ).unwrap();

        let expected_shapes = vec![
            vec![39, 1],
            vec![61, 1],
            vec![50, 1]
        ];

        let mut idx = 0;
        let final_centroids = clf.fit();
        let centroids_unique = final_centroids.unique();
        for item in &centroids_unique {
            let idxs = final_centroids.value_indices(*item);
            let y_train_idxs = y_train.indice_query(idxs).unwrap();
            assert_eq!(
                y_train_idxs.shape().values(), 
                expected_shapes[idx]
            );
            idx += 1;
        }
        
    }
    

}
