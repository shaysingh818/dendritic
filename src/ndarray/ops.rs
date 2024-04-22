use crate::ndarray::ndarray::NDArray;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

/// Generic operations performed for NDArray with f64 type
pub trait Ops {

    fn save(&self, filepath: &str) -> std::io::Result<()>; 
    fn load(filepath: &str) -> std::io::Result<NDArray<f64>>;
    fn apply(&self, loss_func: fn(value: f64) -> f64) -> Result<NDArray<f64>, String>;  
    fn mult(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String>; 
    fn add(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String>; 
    fn subtract(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn dot(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn scale_add(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn transpose(self) -> Result<NDArray<f64>, String>;
    fn permute(self, indice_order: Vec<usize>) -> Result<NDArray<f64>, String>;  
    fn norm(&self, p: usize) -> Result<NDArray<f64>, String>;
    fn mean(&self) -> Result<f64, String>;
    
    // scalar ops
    fn scalar_subtract(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_mult(&self, scalar: f64) -> Result<NDArray<f64>, String>;
    fn scalar_add(&self, scalar: f64) -> Result<NDArray<f64>, String>; 
}

impl Ops for NDArray<f64> {

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
        let mut result = NDArray::new(self.shape().to_vec()).unwrap(); 
        for x in self.values() {
            let loss_val = loss_func(*x); 
            let _ = result.set_idx(index, loss_val);
            index += 1;  
        }
        Ok(result)
    }

    /// Add two NDArray's and get resulting NDArray instance
    fn add(&self, value: NDArray<f64>) -> Result<NDArray<f64>, String> {

        /* rank mismatch */
        if self.rank() != value.rank() {
            return Err("Add: Rank Mismatch".to_string());
        }


        let curr_shape: &Vec<usize> = self.shape();
        let mut result = NDArray::new(curr_shape.to_vec()).unwrap();
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
            return Err("Add: Rank Mismatch".to_string());
        }

        let curr_shape: &Vec<usize> = self.shape();
        let mut result = NDArray::new(curr_shape.to_vec()).unwrap();
        if self.size() != other.values().len() {
            return Err("Add: Size mismatch for arrays".to_string());
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

        let curr_shape: &Vec<usize> = self.shape();
        let mut result = NDArray::new(curr_shape.to_vec()).unwrap();
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

    /// Perform dot product of current NDArray on another NDArray instance
    fn dot(&self, value: NDArray<f64>) -> Result<NDArray<f64>, String> {

        /* rank mismatch */
        if self.rank() != value.rank() {
            return Err("Dot: Rank Mismatch".to_string());
        }

        if self.rank() != 2 {
            return Err("Dot: Requires rank 2 values".to_string());
        }

        if self.shape()[self.rank()-1] != value.shape()[0] {
            return Err("Dot: Rows must equal columns".to_string());
        }

        let new_shape: Vec<usize> = vec![self.shape()[0], value.shape()[self.rank()-1]];
        let mut result = NDArray::new(new_shape).unwrap();

        /* stride values to stay in constant time */ 
        // let mut counter = 0; 
        let mut row_counter = 0; 
        let mut col_counter = 0; 
        let mut stride = 0;  
        for counter in 0..result.size() {

            if stride == value.shape()[self.rank()-1]  {
                row_counter += 1;
                stride = 0; 
            }

            if col_counter >= value.shape()[value.rank()-1]-1 {
                col_counter = 0; 
            }

            let curr = self.rows(row_counter).unwrap();
            let val = value.cols(col_counter).unwrap();

            /* multiply */ 
            let mut value = 0.0; 
            for item in 0..curr.len() {
                value += curr[item] * val[item];
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

        let value_shape = value.shape();
        if value_shape[0] != 1 {
            return Err("Scale add must have a vector dimension (1, N)".to_string());
        }

        let mut total_counter = 0; 
        let mut counter = 0;
        let vector_values = value.values();
        let curr_shape: &Vec<usize> = self.shape();
        let mut result = NDArray::new(curr_shape.to_vec()).unwrap();
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

    /// Tranpose current NDArray instance, works only on rank 2 values
    fn transpose(self) -> Result<NDArray<f64>, String> {

        if self.rank() != 2 {
            return Err("Transpose must contain on rank 2 values".to_string());
        }

        let mut index = 0;
        let shape = self.shape();
        let mut reversed_shape = shape.clone();  
        reversed_shape.reverse();
        let mut result = NDArray::new(reversed_shape.to_vec()).unwrap();

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
        let curr_shape = self.shape();
        let mut new_shape = Vec::new();
        for item in &indice_order {
            new_shape.push(curr_shape[*item]);
        }

        let mut result = NDArray::new(new_shape).unwrap();
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

        let mut result = NDArray::new(self.shape().to_vec()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index]; 
            let raised = value.powf(p as f64); 
            let _ = result.set_idx(index, raised);
        }
        Ok(result)
    }


    fn mean(&self) -> Result<f64, String> {
        let mut total = 0.0; 
        for index in 0..self.size() {
            let value = self.values()[index]; 
            total += value; 
        }
        Ok(total / self.size() as f64)
    } 


    fn scalar_subtract(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().to_vec()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] - scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    fn scalar_add(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().to_vec()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] + scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }


    fn scalar_mult(&self, scalar: f64) -> Result<NDArray<f64>, String> {
        let mut result = NDArray::new(self.shape().to_vec()).unwrap();
        for index in 0..self.size() {
            let value = self.values()[index] * scalar; 
            let _ = result.set_idx(index, value);
        }
        Ok(result)
    }

}