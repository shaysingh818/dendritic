use crate::ndarray::NDArray;
use std::collections::btree_map::BTreeMap;
use itertools::Itertools;


pub trait UnaryOps {
    fn transpose(self) -> Result<NDArray<f64>, String>;
    fn permute(self, indice_order: Vec<usize>) -> Result<NDArray<f64>, String>; 
    fn norm(&self, p: usize) -> Result<NDArray<f64>, String>;
    fn signum(&self) -> Result<NDArray<f64>, String>;
    fn sum_axis(&self, axis: usize) -> Result<NDArray<f64>, String>;
    fn select_axis(&self, axis: usize, indices: Vec<usize>) -> Result<NDArray<f64>, String>;
    fn apply(&self, loss_func: fn(value: f64) -> f64) -> Result<NDArray<f64>, String>;
    fn argmax(&self, axis: usize) -> NDArray<f64>;
}


impl UnaryOps for NDArray<f64> {

    /// Tranpose current NDArray instance, works only on rank 2 values
    fn transpose(self) -> Result<NDArray<f64>, String> {

        if self.rank() != 2 {
            return Err("Transpose must contain on rank 2 values".to_string());
        }

        let mut index = 0;
        let mut result = NDArray::new(self.shape().reverse()).unwrap();

        for _item in self.values() {

            let indices = self.indices(index).unwrap();
            let mut reversed_indices = indices.clone();
            reversed_indices.reverse();

            let idx = self.index(indices).unwrap();
            let val = self.values()[idx]; 

            /* set value from reversed */ 
            let _ = result.set(reversed_indices ,val);
            index += 1; 
        }

        Ok(result)

    }

    /// Permute indices of NDArray. Can be used to peform transposes/contraction on rank 3 or higher values.
    fn permute(self, indice_order: Vec<usize>) -> Result<NDArray<f64>, String> {

        if indice_order.len() != self.rank() {
            return Err("Indice order must be same length as rank".to_string());
        }

        let mut index = 0;
        let permuted_shape = self.shape().permute(indice_order.clone());
        let mut result = NDArray::new(permuted_shape).unwrap();
        for _item in self.values() {

            let indices = self.indices(index).unwrap();
            let mut new_indice_order = Vec::new();
            for item in &indice_order {
                new_indice_order.push(indices[*item])
            }

            let idx = self.index(indices.clone()).unwrap();
            let val = self.values()[idx]; 

            /* set value from reversed */ 
            let _ = result.set(new_indice_order ,val);
            index += 1; 
        }

        Ok(result)
    }


    /// L2 norm can also be  x^t x
    fn norm(&self, p: usize) -> Result<NDArray<f64>, String> {

        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index]; 
            let raised = value.powf(p as f64); 
            let _ = result.set_idx(index, raised);
        }
        Ok(result)
    }

    
    /// Adds values based on x < 0 < 1
    fn signum(&self) -> Result<NDArray<f64>, String> {

        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index]; 
            if value < 0.0 {
                let _ = result.set_idx(index, -1.0);
            } else if value > 0.0 {
                let _ = result.set_idx(index, 1.0);
            } else { 
                let _ = result.set_idx(index, 0.0);
            }
        }

        Ok(result)
    }


    fn sum_axis(&self, axis: usize) -> Result<NDArray<f64>, String> {

        if axis > self.rank()-1 {
            return Err("Sum Axis: Axis greater than rank".to_string())
        }

        if self.rank() > 2 {
            return Err("Sum Axis: Not supported for rank 2 or higher values yet".to_string());
        }

        if axis == 0 {
            let mut result = NDArray::new(vec![1,1]).unwrap();
            let sum: f64 = self.values().iter().sum();
            let _ = result.set_idx(0, sum);
            return Ok(result);
        }


        let sum_stride = self.size() / self.shape().dim(axis);
        let axis_stride = self.shape().dim(axis.clone());
        let result_shape: Vec<usize> = vec![axis, axis_stride];
        let mut result = NDArray::new(result_shape.clone()).unwrap();

        let mut idx = 0; 
        let mut sum: f64 = 0.0; 
        let mut stride_counter = 0; 
        for item in self.values() {

            if stride_counter == sum_stride {
                let _ = result.set_idx(idx, sum);  
                stride_counter = 0;
                sum = 0.0;
                idx += 1;  
            }

            sum += item;
            stride_counter += 1;
        }

        let _ = result.set_idx(idx, sum); 
        Ok(result)
    }


    /// Select specific indices from an axis
    fn select_axis(&self, axis: usize, indices: Vec<usize>) -> Result<NDArray<f64>, String> {
 
        if axis > self.rank() - 1 { 
            return Err(
                "Axis Indices: Selected axis larger than rank".to_string()
            );
        }

        if self.rank() > 2 {
            return Err(
                "Select Axis: Only works on rank 2 values and lower".to_string()
            );

        }

        let mut curr_shape = self.shape().values(); 
        curr_shape[axis] = indices.len();

        let mut result: NDArray<f64> = NDArray::new(
            curr_shape.clone()
        ).unwrap();

        for (index, indice) in indices.iter().enumerate() {
            let axis_vals = self.axis(axis, *indice).unwrap();
            let mut counter = 0; 
            for (idx, val) in axis_vals.values().iter().enumerate() {
                let remainder_idx = self.rank() - 1 - axis;
                let mut indices: Vec<usize> = vec![0; self.rank()];
                indices[axis] = index; 
                indices[remainder_idx] = idx; 
                result.set(indices, *val); 
            }
        }

        Ok(result)
    }


    /// Apply loss function on values in ndarray
    fn apply(&self, loss_func: fn(value: f64) -> f64) -> Result<NDArray<f64>, String> {   
        let mut index = 0; 
        let mut result = NDArray::new(self.shape().values()).unwrap(); 
        for x in self.values() {
            let loss_val = loss_func(*x); 
            let _ = result.set_idx(index, loss_val);
            index += 1;  
        }
        Ok(result)
    }

    fn argmax(&self, axis: usize) -> NDArray<f64> {

        // this only works for a row (for now)
        let mut results = Vec::new();
        let shape = self.shape().dim(axis);
        for idx in 0..shape {
            let axis_value = self.axis(axis, idx).unwrap();

            let mut curr_max = 0.0; 
            let mut index = 0; 
            let mut final_index = 0;
            for item in axis_value.values() {
                if item > &curr_max {
                    curr_max = *item;
                    final_index = index; 
                }
                index += 1;
            }

            results.push(final_index as f64);
        }

        let result = NDArray::array(
            vec![shape, 1],
            results
        ).unwrap();
        result

    }


}
