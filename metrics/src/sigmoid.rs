use ndarray::ndarray::NDArray;


pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
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


pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        return x;
    }
    0.0
}
