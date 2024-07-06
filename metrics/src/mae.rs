use ndarray::ndarray::NDArray;


pub fn mae(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<f64, String>  {

    if y_true.size() != y_pred.size() {
        return Err("Size of y values do not match".to_string());
    }

    let mut index = 0;
    let mut result = 0.0;  
    for item in y_true.values() {
        let diff = item - y_pred.values()[index];
        result += diff.abs();
        index += 1; 
    }

    result = result * 1.0/y_true.size() as f64; 
    Ok(result)
}
