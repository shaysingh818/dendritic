use dendritic_ndarray::ndarray::NDArray;

/// Mean squared error function
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

/// Binary cross entropy for logistic binary classification
pub fn binary_cross_entropy(y_hat: &NDArray<f64>, y_true: &NDArray<f64>) -> Result<f64, String> {
    
    let mut index = 0;
    let mut result = 0.0;  
    for y in y_true.values() {
      
        let y_pred = y_hat.values()[index];
        let diff = y * y_pred.ln() + (1.0 - y) * (1.0-y_pred).ln();
        result += diff;
        index += 1;
    }

    result = -(1.0/y_hat.size() as f64) * result;
    Ok(result)
}

/// Categorical cross entropy for multi class classification
pub fn categorical_cross_entropy(y_hat: &NDArray<f64>, y_true: &NDArray<f64>) -> Result<f64, String> {

    let mut index = 0;
    let mut result = 0.0;  
    for y in y_true.values() {
        let y_pred = y_hat.values()[index];
        let diff = -y * y_pred.ln();
        result += diff;
        index += 1;
    }

    Ok(result * 1.0/y_hat.size() as f64)

}

