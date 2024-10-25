use ndarray::ndarray::NDArray;
use ndarray::ops::*;


/// Get class indices of target variable in dataset
pub fn class_idxs(target: &NDArray<f64> ) -> Vec<Vec<usize>> {
    let mut class_indices: Vec<Vec<usize>> = Vec::new();
    let y_target = target.unique();
    for item in &y_target {
        let indices = target.value_indices(*item);
        class_indices.push(indices);
    }

    class_indices
}

/// Get likelihood of class probability with target and class indexes
pub fn class_probabilities(
    target: &NDArray<f64>,
    class_idxs: Vec<Vec<usize>>) -> Vec<f64> {
    let mut probabilities: Vec<f64> = Vec::new();
    for item in &class_idxs {
        let val = item.len() as f64 /target.size() as f64;
        probabilities.push(val as f64);
    }
    probabilities
}

/// Gaussian probability of row sample
pub fn gaussian_probability(x: f64, mu: f64, sigma: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let denom = sigma * (2.0 * pi).sqrt();
    let second = -((x - mu).powf(2.0) / (2.0 * sigma.powf(2.0)));
    let e_second = second.exp();
    (1.0 / denom) * e_second
}
