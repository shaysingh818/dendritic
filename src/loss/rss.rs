use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::Ops;

pub fn residual_sum_squares(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<f64, String> {

    if y_true.size() != y_pred.size() {
        return Err("Size of y values do not match".to_string());
    }
 
    let subtract = y_true.subtract(y_pred).unwrap();
    let mut loss = 0.0; 
    for item in subtract.values() {
        loss += item.powf(2.0); 
    }
    Ok(loss)
}
