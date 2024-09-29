use ndarray::ndarray::NDArray;
use ndarray::ops::*;


#[derive(Debug)]
pub struct KMeans {
    pub data: NDArray<f64>,
    pub k: usize,
    pub centroids: Vec<NDArray<f64>>,
    distance_metric: fn(
        y1: &NDArray<f64>, 
        y2: &NDArray<f64>
    ) -> Result<f64, String>
}


impl KMeans {

    pub fn fit(
        data: &NDArray<f64>, 
        k: usize,
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
            centroids: centroids, 
            distance_metric: distance_metric
        })
    }


    pub fn assign_clusters(&self) {
        
        let samples = self.data.shape().dim(0);
        for row in 0..samples {
            let mut clusters: Vec<f64> = Vec::new();
            let item = self.data.axis(0, row).unwrap();
            for centroid in &self.centroids {
                let dist = (self.distance_metric)(
                    &item, 
                    centroid
                ).unwrap();
                clusters.push(dist);
            }
            //println!("{:?}", clusters);

            let min_idx = clusters.iter().map(|p| p).min().unwrap_or(0.0);
            println!("{:?}", min_idx); 


        }

    }


    

}
