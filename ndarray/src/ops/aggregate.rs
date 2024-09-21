use crate::ops::unary::UnaryOps; 
use crate::ops::scalar::ScalarOps; 
use crate::ndarray::NDArray;
use std::collections::btree_map::BTreeMap;
use itertools::Itertools;

pub trait AggregateOps {
    fn avg(&self) -> f64;
    fn length(&self) -> f64;
    fn square(&self) -> Result<NDArray<f64>, String>;
    fn sum(&self) -> Result<NDArray<f64>, String>; 
    fn abs(&self) -> Result<NDArray<f64>, String>;
    fn sort(&self) -> Vec<f64>;
    fn unique(&self) -> Vec<f64>;
    fn mean(&self, axis: usize) -> Result<Vec<f64>, String>;
    fn stdev(&self, axis: usize) -> Result<Vec<f64>, String>; 
    fn stdev_sample(&self, axis: usize) -> Result<Vec<f64>, String>;
}


impl AggregateOps for NDArray<f64> {

    /// Take the average of all elements in ndarray structure
    fn avg(&self) -> f64 {
        let sum: f64 = self.values().iter().sum();
        sum / self.size() as f64
    }

    /// Computes the length or magnitude of vector
    fn length(&self) -> f64 {
        let mut sum: f64 = 0.0;
        for index in 0..self.size() {
            let value = self.values()[index]; 
            let raised = value.powf(2.0);
            sum += raised;
        }

        sum.sqrt()
    }


    /// Raise all elements to the second power
    fn square(&self) -> Result<NDArray<f64>, String> {

        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index]; 
            let raised = value.powf(2.0); 
            let _ = result.set_idx(index, raised);
        }

        Ok(result)
    }


    /// sum all elements in ndarray structure
    fn sum(&self) -> Result<NDArray<f64>, String> {

        let sum_val = self.values().iter().sum();
        let result = NDArray::array(
            vec![1, 1],
            vec![sum_val]
        ).unwrap();

        Ok(result)

    }


    /// Get the absolute value of each element in ndarray
    fn abs(&self) -> Result<NDArray<f64>, String> {

        let abs: Vec<f64> = self.values().into_iter().map(
            |val| val.abs()
        ).collect();

        let result = NDArray::array(
            self.shape().values(), abs
        ).unwrap();

        Ok(result)
    }


    /// Sort values in ndarray on specific axis
    fn sort(&self) -> Vec<f64> {
        let mut values = self.values().clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.to_vec()
    }


    /// Get unique values in ndarray
    fn unique(&self) -> Vec<f64> {
        let mut values = self.values().clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        values.to_vec()
    }


    /// Calculate the mean of values along a specific axis
    fn mean(&self, axis: usize) -> Result<Vec<f64>, String> {

        let mut results: Vec<f64> = Vec::new(); 
        let shape_axis = self.shape().dim(axis);
        for item in 0..shape_axis {
            let axis_vals = self.axis(axis, item).unwrap();  
            let sum_vals: f64 = axis_vals.values().iter().sum(); 
            let avg: f64 = sum_vals / axis_vals.values().len() as f64;
            results.push(avg); 
        }

        Ok(results)
    }

    /// Calculate the mean of values along a specific axis
    fn stdev(&self, axis: usize) -> Result<Vec<f64>, String> {

        if axis >= self.shape().values().len() {
            let msg = "stdev: Axis too large for current array";
            return Err(msg.to_string());
        }

        let mut results: Vec<f64> = Vec::new(); 
        let mean_axis = self.mean(axis).unwrap();
        let shape_axis = self.shape().dim(axis);
        for shape in 0..shape_axis {
            let axis_vals = self.axis(axis, shape).unwrap();
            let mean_axis_val = mean_axis[shape];
            let subtract_mean = axis_vals.scalar_subtract(mean_axis_val).unwrap();
            let squared = subtract_mean.norm(2).unwrap();
            let sum: f64 = squared.values().iter().sum();
            let avg = sum / axis_vals.size() as f64;
            results.push(avg.sqrt()); 
        }

        Ok(results)
    }


    /// Calculate the mean of values along a specific axis
    fn stdev_sample(&self, axis: usize) -> Result<Vec<f64>, String> {

        if axis >= self.shape().values().len() {
            let msg = "stdev sample: Axis too large for current array";
            return Err(msg.to_string());
        }

        let mut results: Vec<f64> = Vec::new(); 
        let mean_axis = self.mean(axis).unwrap();
        let shape_axis = self.shape().dim(axis);
        for shape in 0..shape_axis {
            let axis_vals = self.axis(axis, shape).unwrap();
            let mean_axis_val = mean_axis[shape];
            let subtract_mean = axis_vals.scalar_subtract(mean_axis_val).unwrap();
            let squared = subtract_mean.norm(2).unwrap();
            let sum: f64 = squared.values().iter().sum();
            let avg = sum / (axis_vals.size() - 1) as f64;
            results.push(avg.sqrt()); 
        }

        Ok(results)
    }


}
