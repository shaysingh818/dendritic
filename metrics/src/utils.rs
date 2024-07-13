use ndarray::ndarray::NDArray;
use ndarray::ops::*; 

pub fn apply(
    value: NDArray<f64>,
    axis: usize, 
    loss_function: fn(value: NDArray<f64>) -> NDArray<f64>) -> NDArray<f64> {

    let mut feature_vec: Vec<f64> = Vec::new();
    let shape = value.shape().dim(axis);
    for idx in 0..shape {
        let axis_value = value.axis(axis, idx).unwrap();
        let axis_loss = (loss_function)(axis_value);
        let mut axis_applied = axis_loss.values().clone(); 
        feature_vec.append(&mut axis_applied);
    }

    let result: NDArray<f64> = NDArray::array(
        value.shape().values(),
        feature_vec.clone()
    ).unwrap();

    result
}


