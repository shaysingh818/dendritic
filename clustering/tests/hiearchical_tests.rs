
#[cfg(test)]
mod hierarchical_clustering_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use knn::distance::*;
    use clustering::hierarchical::*;

    #[test]
    fn test_hierarchical_cluster() {

        let data_path = "data/hierarchical/data";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();

        assert_eq!(
            data.shape().values(),
            vec![6, 2]
        );

        let clf = HierarchicalClustering::new(&data, euclidean).unwrap();
        assert_eq!(
            clf.data.shape().values(),
            vec![6, 2]
        );
        
        assert_eq!(
            clf.distance_matrix.shape().values(),
            vec![6, 6]
        ); 

        let bd_data = NDArray::new(vec![1, 1]).unwrap();
        let clf_bad = HierarchicalClustering::new(&bd_data, euclidean);
        assert_eq!(
            clf_bad.unwrap_err(),
            "Not enough rows in sample data"
        ); 

    }


    #[test]
    fn test_calculate_distance_matrix() {

        let data_path = "data/hierarchical/data";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();

        assert_eq!(
            data.shape().values(),
            vec![6, 2]
        );

        let mut clf = HierarchicalClustering::new(
            &data, 
            euclidean
        ).unwrap();

        assert_eq!(
            clf.data.shape().values(),
            vec![6, 2]
        );

        let expected_count = vec![6, 5, 4, 3, 2, 1];

        clf.calculate_distance_matrix();

        assert_eq!(
            clf.distance_matrix.shape().values(),
            vec![6, 6]
        );

        let dist_mtx = clf.distance_matrix();
        for col in 0..dist_mtx.shape().dim(0) {
            let item = dist_mtx.axis(0, col).unwrap();
            let vals = item.values();
            let count = vals.iter().filter(|&n| *n == 0.0).count(); 
            assert_eq!(
                count,
                expected_count[col]
            ); 
        }
 
    }


    #[test]
    fn test_fit_transform() {

        let data2: NDArray<f64> = NDArray::array(
            vec![5, 2],
            vec![
                0.07, 0.83, 
                0.85, 0.14, 
                0.66, 0.89,
                0.49, 0.64,
                0.80, 0.46
            ]
        ).unwrap();

        let data_path = "data/hierarchical/data";
        let data: NDArray<f64> = NDArray::load(data_path).unwrap();

        assert_eq!(
            data.shape().values(),
            vec![6, 2]
        );

        let mut clf = HierarchicalClustering::new(
            &data2, 
            euclidean
        ).unwrap();

        assert_eq!(
            clf.data.shape().values(),
            vec![5, 2]
        );

        //clf.fit_transform();

    } 

    #[test]
    fn test_cluster_min_dist() {

        let data: NDArray<f64> = NDArray::array(
            vec![5, 2],
            vec![
                0.07, 0.83, 
                0.85, 0.14, 
                0.66, 0.89,
                0.49, 0.64,
                0.80, 0.46
            ]
        ).unwrap();


        let data_path = "data/hierarchical/data";
        let data2: NDArray<f64> = NDArray::load(data_path).unwrap();

        let mut clf = HierarchicalClustering::new(
            &data, 
            euclidean
        ).unwrap();

        // first iteration
        clf.calculate_distance_matrix();
        let dist_mat = clf.distance_matrix();
        let coord = clf.find_min_coord(&dist_mat);
        let new_mat = clf.update_dist_mat(dist_mat.clone(), &coord);

        let coord_2 = clf.find_min_coord(&new_mat);
        let new_mat_2 = clf.update_dist_mat(new_mat.clone(), &coord_2);
 
        for row in 0..new_mat_2.shape().dim(0) {
            let item = new_mat_2.axis(0, row).unwrap();
            println!("{:?}", item.values()); 
        } 





        //println!("{:?}", clf.clusters()); 






    
    }

}
