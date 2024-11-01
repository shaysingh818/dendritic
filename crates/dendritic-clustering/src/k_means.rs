use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;


#[derive(Debug)]
pub struct KMeans {
    pub data: NDArray<f64>,
    pub k: usize,
    pub max_iter: usize, 
    pub centroids: Vec<NDArray<f64>>,
    distance_metric: fn(
        y1: &NDArray<f64>, 
        y2: &NDArray<f64>
    ) -> Result<f64, String>
}


impl KMeans {

    /// Create new instance of K Means clustering model
    pub fn new(
        data: &NDArray<f64>, 
        k: usize,
        max_iter: usize,
        distance_metric: fn(
            y1: &NDArray<f64>, 
            y2: &NDArray<f64>) -> Result<f64, String>
        ) -> Result<KMeans, String> {

        let rows = data.shape().dim(0);
        if rows <= k {
            return Err(
                "Not enough rows in sample data".to_string()
            );
        }

        // select first N rows for centroids
        let mut centroids: Vec<NDArray<f64>> = Vec::new();
        for n in 0..k {
            let row = data.axis(0, n).unwrap();
            centroids.push(row);
        }


        Ok(Self {
            data: data.clone(),
            k: k,
            max_iter: max_iter, 
            centroids: centroids, 
            distance_metric: distance_metric
        })
    }

    
    /// Retrieve centroids of K means clustering
    pub fn centroids(&self) -> &Vec<NDArray<f64>> {
        &self.centroids
    }

    /// Set centroids of K Means clustering
    pub fn set_centroids(&mut self, indices: &Vec<usize>) {
        let mut centroids: Vec<NDArray<f64>> = Vec::new();
        for n in indices {
            let row = self.data.axis(0, *n).unwrap();
            centroids.push(row);
        }
        self.centroids = centroids;
    }

    /// Assign clusters based on distance metric
    pub fn assign_clusters(&self) -> NDArray<f64> {

        let mut total: Vec<f64> = Vec::new();
        let samples = self.data.shape().dim(0);
        for row in 0..samples {
            let item = self.data.axis(0, row).unwrap();
            for centroid in &self.centroids {
                let dist = (self.distance_metric)(
                    &item, 
                    centroid
                ).unwrap();
                total.push(dist);
            }
        }

        NDArray::array(
            vec![samples, self.k],
            total
        ).unwrap().argmin(0).unwrap()
    }

    /// Calculate centroids for K means clustering model
    pub fn calculate_centroids(&mut self, values: &NDArray<f64>) {

        let mut cluster_idxs: Vec<Vec<usize>> = Vec::new();
        let categories = values.unique();
        for category in &categories {
            let idxs = values.value_indices(*category);
            cluster_idxs.push(idxs); 
        }

        /* first cluster */
        let mut centroids = Vec::new();
        for item in &cluster_idxs { 
            let vals = self.data.axis_indices(0, item.clone()).unwrap();
            let mean_vals = vals.mean(1).unwrap();
            let nd: NDArray<f64> = NDArray::array(
                vec![mean_vals.len(), 1],
                mean_vals
            ).unwrap();
            centroids.push(nd);
        }

        self.centroids = centroids;
    }

    /// Fit clusters for K means
    pub fn fit(&mut self) -> NDArray<f64> {

        let mut assigned_clusters = self.assign_clusters();
        for _epoch in 0..self.max_iter {
            self.calculate_centroids(&assigned_clusters);
            assigned_clusters = self.assign_clusters();
        }

        assigned_clusters
    }

}
