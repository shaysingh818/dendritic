use crate::ndarray::NDArray;
use std::fs::File;
use std::io::{BufWriter, Read, Write}; 

pub trait BinaryOps {
    fn mult(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String>; 
    fn add(&self, other: NDArray<f64>) -> Result <NDArray<f64 >, String>;
    fn subtract(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn dot(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn scale_add(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn scale_mult(&self, other: NDArray<f64>) -> Result<NDArray<f64>, String>;
    fn save(&self, filepath: &str) -> std::io::Result<()>; 
    fn load(filepath: &str) -> std::io::Result<NDArray<f64>>;
}


impl BinaryOps for NDArray<f64> {


    /// Multiply an ndarray by another
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

    /// Elementwise multiplication of ndarray
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
}
