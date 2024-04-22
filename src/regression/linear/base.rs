use crate::ndarray::ndarray::NDArray;


pub trait Linear {
    fn predict(&mut self, inputs: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn forward(&self) -> Result<NDArray<f64>, String>;
    fn weight_update(&mut self, y_pred: NDArray<f64>);
    fn bias_update(&mut self, y_pred: NDArray<f64>);
    fn set_loss(&mut self, loss_func: fn(y_true: NDArray<f64>, y_pred: NDArray<f64>) -> Result<f64, String>);
    fn train(&mut self, epochs: usize, log_output: bool);
}