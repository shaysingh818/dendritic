use ndarray::ndarray::NDArray;
use ndarray::ops::*;


pub fn calculate_distances(
    distance_metric: fn(
        y1: &NDArray<f64>, 
        y2: &NDArray<f64>) -> Result<f64, String>,
    features: &NDArray<f64>,
    point: &NDArray<f64>) -> Result<Vec<(f64, usize)>, String> {
    
    let rows = features.shape().dim(0);
    let pt_rows = point.shape().dim(0);

    if pt_rows != features.shape().dim(1) {
        let msg = "KNN: Rows of point doesn't match cols of sample data";
        return Err(msg.to_string());
    }
    
    let mut distances: Vec<(f64, usize)> = Vec::new();
    for row in 0..rows {
        let item = features.axis(0, row).unwrap();
        let dist = (distance_metric)(&point, &item).unwrap();
        distances.push((dist, row));
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(distances.to_vec())
}
