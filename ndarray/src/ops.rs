use crate::ndarray::NDArray;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::collections::btree_map::BTreeMap;
use itertools::Itertools;


/// Generic operations performed for NDArray with f64 type
pub trait Ops {

    // mutator ops
    fn square(&self) -> Result<NDArray<f64>, String>;
    fn sum(&self) -> Result<NDArray<f64>, String>; 
    fn abs(&self) -> Result<NDArray<f64>, String>;
    fn signum(&self) -> Result<NDArray<f64>, String>;

    fn sort(&self) -> Vec<f64>;
    fn unique(&self) -> Vec<f64>;
    fn save(&self, filepath: &str) -> std::io::Result<()>; 
    fn load(filepath: &str) -> std::io::Result<NDArray<f64>>;
    fn apply(&self, loss_func: fn(value: f64) -> f64) -> Result<NDArray<f64>, String>; 
    fn mult(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String>; 
    fn add(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String>;
    fn sum_axis(&self, axis: usize) -> Result<NDArray<f64>, String>; 
    fn subtract(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn dot(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn scale_add(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn mean(&self, axis: usize) -> Result<Vec<f64>, String>;
    fn stdev(&self, axis: usize) -> Result<Vec<f64>, String>;
    fn argmax(&self, axis: usize) -> NDArray<f64>;

    fn scale_mult(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn transpose(self) -> Result<NDArray<f64>, String>;
    fn permute(self, indice_order: Vec<usize>) -> Result<NDArray<f64>, String>;  
    fn norm(&self, p: usize) -> Result<NDArray<f64>, String>;
    
    // scalar ops
    fn avg(&self) -> f64; 
    fn scalar_subtract(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_mult(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_add(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_div(&self, scalar: f64) -> Result<NDArray<f64>, String>; 

}

impl Ops for NDArray<f64> {

    /// Get unique values in ndarray
    fn unique(&self) -> Vec<f64> {
        let mut values = self.values().clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        values.to_vec()
    }


    /// Sort values in ndarray on specific axis
    fn sort(&self) -> Vec<f64> {
        let mut values = self.values().clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.to_vec()
    }

    /// Save instance of NDArray to json file with serialized values
    fn save(&self, filepath: &str) -> std::io::Result<()> {
        let filename_format = format!("{filepath}.json");
        let file = match File::create(filename_format) {
            Ok(file) => file,
            Err(err) => {
                return Err(err);
            }
        };
        let mut writer = BufWriter::new(file);
        let json_string = serde_json::to_string_pretty(&self)?;
        writer.write_all(json_string.as_bytes())?;
        Ok(())
    }


    /// Load Instance of saved NDarray, serialize to NDArray structure
    fn load(filepath: &str) -> std::io::Result<NDArray<f64>> {
        let filename_format = format!("{filepath}.json");
        let mut file = match File::open(filename_format) {
            Ok(file) => file,
            Err(err) => {
                return Err(err);
            }
        };
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let instance: NDArray<f64> = serde_json::from_str(&contents)?;
        Ok(instance)
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


    fn square(&self) -> Result<NDArray<f64>, String> {

        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index]; 
            let raised = value.powf(2.0); 
            let _ = result.set_idx(index, raised);
        }

        Ok(result)
    }


    fn sum(&self) -> Result<NDArray<f64>, String> {

        let sum_val = self.values().iter().sum();
        let result = NDArray::array(
            vec![1, 1],
            vec![sum_val]
        ).unwrap();

        Ok(result)

    }


    fn abs(&self) -> Result<NDArray<f64>, String> {

        let abs: Vec<f64> = self.values().into_iter().map(
            |val| val.abs()
        ).collect();

        let result = NDArray::array(
            self.shape().values(), abs
        ).unwrap();

        Ok(result)
    }


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


    /// Add two NDArray's and get resulting NDArray instance
    fn add(&self, value: NDArray<f64>) -> Result<NDArray<f64>, String> {

        /* rank mismatch */
        if self.rank() != value.rank() {
            return Err("Add: Rank Mismatch".to_string());
        }

        let mut result = NDArray::new(self.shape().values()).unwrap();
        if self.size() != value.values().len() {
            return Err("Add: Size mismatch for arrays".to_string());
        }

        let mut counter = 0; 
        let values = value.values(); 
        for item in self.values() {
            let add_result = item + values[counter];
            let _ = result.set_idx(counter, add_result);
            counter += 1;
        }

        Ok(result)
    }


    fn mult(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String> {

        /* rank mismatch */
        if self.rank() != other.rank() {
            return Err("Mult: Rank Mismatch".to_string());
        }

        let mut result = NDArray::new(self.shape().values()).unwrap();
        if self.size() != other.values().len() {
            println!("{:?} {:?}", self.size(), other.values().len()); 
            return Err("Mult: Size mismatch for arrays".to_string());
        }

        let mut counter = 0; 
        let values = other.values(); 
        for item in self.values() {
            let mult_result = item * values[counter];
            let _ = result.set_idx(counter, mult_result);
            counter += 1;
        }

        Ok(result)
    } 

    /// Subtract values in NDArray instances
    fn subtract(&self, value: NDArray<f64>) -> Result<NDArray<f64>, String> {

        /* rank mismatch */
        if self.rank() != value.rank() {
            return Err("Subtract: Rank Mismatch".to_string());
        }

        let mut result = NDArray::new(self.shape().values()).unwrap();
        if self.size() != value.values().len() {
            return Err("Subtract: Size mismatch for arrays".to_string());
        }

        let mut counter = 0; 
        let values = value.values(); 
        for item in self.values() {
            let add_result = item - values[counter];
            let _ = result.set_idx(counter, add_result);
            counter += 1;
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


    /// Perform dot product of current NDArray on another NDArray instance
    fn dot(&self, input: NDArray<f64>) -> Result<NDArray<f64>, String> {

        /* rank mismatch */
        if self.rank() != input.rank() {
            return Err("Dot: Rank Mismatch".to_string());
        }

        if self.rank() != 2 {
            return Err("Dot: Requires rank 2 values".to_string());
        }

        if self.shape().dim(self.rank()-1) != input.shape().dim(0) {
            return Err("Dot: Rows must equal columns".to_string());
        }

        let new_shape: Vec<usize> = vec![self.shape().dim(0), input.shape().dim(self.rank()-1)];
        let mut result = NDArray::new(new_shape).unwrap();

        /* stride values to stay in constant time */ 
        // let mut counter = 0; 
        let mut row_counter = 0; 
        let mut col_counter = 0; 
        let mut stride = 0;  
        for counter in 0..result.size() {

            if stride == input.shape().dim(self.rank()-1)  {
                row_counter += 1;
                stride = 0; 
            }

            let col_dim = input.shape().dim(input.rank()-1);
            if col_counter == col_dim {
                col_counter = 0; 
            }

            let curr: NDArray<f64> = self.axis(0, row_counter).unwrap();
            let val: NDArray<f64> = input.axis(1, col_counter).unwrap();

            /* multiply */ 
            let mut value = 0.0; 
            for item in 0..curr.size() {
                value += curr.idx(item) * val.idx(item);
            }
            result.set_idx(counter, value).unwrap(); 

            
            // counter += 1; 
            col_counter += 1;
            stride += 1;  
                    
        }

        Ok(result)
    }

    /// Add values by scalar for current NDArray instance
    fn scale_add(&self, value: NDArray<f64>) -> Result<NDArray<f64>, String> {

        if value.shape().dim(0) != 1 {
            return Err("Scale add must have a vector dimension (1, N)".to_string());
        }

        let mut total_counter = 0; 
        let mut counter = 0;
        let vector_values = value.values();
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for item in self.values() {
            if counter == value.size() {
                counter = 0;
            }
             let add_result = item + vector_values[counter];
             let _ = result.set_idx(total_counter, add_result);
             total_counter += 1; 
        }

        Ok(result)

    }


    fn scale_mult(&self, value: NDArray<f64>) -> Result<NDArray<f64>, String> {
    
        let value_shape = value.shape();
        if value_shape.dim(0) != 1 {
            return Err("Scale add must have a vector dimension (1, N)".to_string());
        }

        let mut total_counter = 0; 
        let mut counter = 0;
        let vector_values = value.values();
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for item in self.values() {
            if counter == value.size() {
                counter = 0;
            }
             let add_result = item * vector_values[counter];
             let _ = result.set_idx(total_counter, add_result);
             total_counter += 1; 
        }

        Ok(result)
    }



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


    // L2 norm can also be  x^t x
    fn norm(&self, p: usize) -> Result<NDArray<f64>, String> {

        // count number of non zero elements

        //

        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index]; 
            let raised = value.powf(p as f64); 
            let _ = result.set_idx(index, raised);
        }
        Ok(result)
    }


    fn avg(&self) -> f64 {
        let sum: f64 = self.values().iter().sum();
        sum / self.size() as f64
    }

    fn scalar_subtract(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] - scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    fn scalar_add(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] + scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    fn scalar_mult(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] * scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    fn scalar_div(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().values()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] / scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }

}


pub trait UtfOps {
    fn unique(&self) -> Vec<&str>;
    fn counts(&self) -> Vec<usize>;
}


impl UtfOps for NDArray<&str> {

    /// Get unique values in ndarray
    fn unique(&self) -> Vec<&str> {
        let values = self.values().clone();
        let unique_vals = values.into_iter().unique().collect();
        unique_vals
    }

    /// Get counts of occurrence for word in string vector
    fn counts(&self) -> Vec<usize> {

        let mut counts: Vec<usize> = Vec::new();
        let mut count = BTreeMap::new();
        for item in self.values() {
            *count.entry(item).or_insert(0) += 1;
        }

        for (_word, count) in & count {
            counts.push(*count);
        }

        counts
    }

}
