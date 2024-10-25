use crate::ndarray::NDArray;
use std::collections::btree_map::BTreeMap;
use itertools::Itertools;


pub trait ScalarOps {
    fn scalar_subtract(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_mult(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_add(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_div(&self, scalar: f64) -> Result<NDArray<f64>, String>; 
}


impl ScalarOps for NDArray<f64> {

    /// Subtract all values in ndarray by scalar
    fn scalar_subtract(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] - scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    /// Add all values in ndarray by scalar
    fn scalar_add(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] + scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    /// Multiply all values in ndarray by scalar
    fn scalar_mult(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] * scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    /// Divide all values in ndarray by scalar
    fn scalar_div(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] / scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }

}
