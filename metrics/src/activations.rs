use ndarray::ndarray::NDArray;


pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}


pub fn sigmoid_vec(x: NDArray<f64>) -> NDArray<f64> {

    let sig = x.values().iter().map(
        |x| sigmoid(*x)
    ).collect(); 

    let result: NDArray<f64> = NDArray::array(
        x.shape().values(),
        sig
    ).unwrap();
    result
}


pub fn sigmoid_prime(x: NDArray<f64>) -> NDArray<f64> {

    let sig = x.values().iter().map(
        |x| sigmoid(*x) * (1.0 - sigmoid(*x))
    ).collect(); 

    let result: NDArray<f64> = NDArray::array(
        x.shape().values(),
        sig
    ).unwrap();
    result
}


pub fn softmax(x: NDArray<f64>) -> NDArray<f64> {

    // get max value from input array
    let max_x: f64 = *x.values().iter().max_by(
        |a, b| a.total_cmp(b)
    ).unwrap();

    let exp_values: Vec<f64> = x.values().iter().map(
        |&x| f64::exp(x - max_x)
    ).collect();

    let sum_exp_vals: f64 = exp_values.iter().sum();
    let mut results: Vec<f64> = Vec::new();

    for val in x.values() {
        let e_x = f64::exp(*val - max_x);
        let val = e_x/sum_exp_vals;
        results.push(val);
    }

    let result: NDArray<f64> = NDArray::array(
        x.shape().values(),
        results
    ).unwrap();

    result
}


pub fn softmax_prime(x: NDArray<f64>) -> NDArray<f64>  {

    let softmax_x = softmax(x.clone());

    let mut index = 0; 
    let mut jacobian: NDArray<f64> = NDArray::new(
        vec![x.shape().dim(0), x.shape().dim(0)]
    ).unwrap();

    for _item in 0..jacobian.size() {

        let val = jacobian.indices(index).unwrap();

        if val[0] == val[1] {
            let i = val[0]; 
            let softmax_val = softmax_x.values()[i]; 
            let result = softmax_val * (1.0 - softmax_val);
            let _ = jacobian.set(val, result);
        } else {
            let softmax_i = softmax_x.values()[val[0]];
            let softmax_j = softmax_x.values()[val[1]];
            let result = -softmax_i * softmax_j;
            let _ =  jacobian.set(val, result);
        } 

        index += 1; 
    }

    jacobian
    
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        return x;
    }
    0.0
}
