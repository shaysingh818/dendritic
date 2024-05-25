use crate::ndarray::ndarray::NDArray;


pub fn mse(y_true: &NDArray<f64>, y_pred: &NDArray<f64>) -> Result<f64, String>  {

    if y_true.size() != y_pred.size() {
        return Err("Size of y values do not match".to_string());
    }

    let mut index = 0;
    let mut result = 0.0;  
    for item in y_true.values() {
        let diff = item - y_pred.values()[index];
        result += diff.powf(2.0);
        index += 1; 
    }

    result = result * 1.0/y_true.size() as f64; 
    Ok(result)
}


pub fn mse_prime(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<NDArray<f64>, String> {

    if y_true.size() != y_pred.size() {
        return Err("Size of y values do not match".to_string());
    }

    let mut index = 0;
    let mut result = NDArray::new(y_true.shape().to_vec()).unwrap();
    for item in y_true.values() {
        let diff = item - y_pred.values()[index];
        let val = diff * 2.0/y_true.size() as f64;
        let _ = result.set_idx(index, val);
        index += 1; 
    }

    Ok(result)
}