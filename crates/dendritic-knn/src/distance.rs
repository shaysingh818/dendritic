use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;

/// Euclidean distance between two points
pub fn euclidean(p1: &NDArray<f64>, p2: &NDArray<f64>) -> Result<f64, String> {
    
    if p1.shape().values() != p2.shape().values() {
        let msg = "Supplied points must be of same shape";
        return Err(msg.to_string());
    }

    let subtract = p1.subtract(p2.clone()).unwrap();
    let power = subtract.norm(2).unwrap(); 
    let sum: f64 = power.values().iter().sum();
    Ok(sum.sqrt())
}



/// Manhattan distance between two points
pub fn manhattan(p1: &NDArray<f64>, p2: &NDArray<f64>) -> Result<f64, String> {
    
    if p1.shape().values() != p2.shape().values() {
        let msg = "Supplied points must be of same shape";
        return Err(msg.to_string());
    }

    let p1_abs = p1.abs().unwrap();
    let p2_abs = p2.abs().unwrap();
    let subtract = p1_abs.subtract(p2_abs.clone()).unwrap();
    let subtract_abs = subtract.abs().unwrap();
    let sum: f64 = subtract_abs.values().iter().sum();
    Ok(sum)
}
