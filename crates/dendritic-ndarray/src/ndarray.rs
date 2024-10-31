use serde::{Serialize, Deserialize};
use crate::shape::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NDArray<T> {
    pub shape: Shape,
    pub size: usize,
    pub rank: usize,
    pub values: Vec<T>
}


impl<T: Default + Clone + std::fmt::Debug + std::cmp::PartialEq> NDArray<T> {

    /// Gets the rank of the current array
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the shape dimensions of the array
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the generic values stored in the array
    pub fn values(&self) -> &Vec<T> {
        &self.values
    }
    
    /// Get the current calculated size of the contigous array
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get generic value from provided indices
    pub fn get(&self, indices: Vec<usize>) -> &T {
        &self.values[self.index(indices).unwrap()]
    }

    /// Get generic value from provided indices
    pub fn idx(&self, index: usize) -> &T {
        &self.values[index]
    }

    /// Lets you change the rank of the current ndarray
    pub fn set_rank(&mut self, new_rank: usize) {
        self.rank = new_rank;
    }

    /// Create instance of NDArray, provide shape dimensions as parameter
    pub fn new(shape: Vec<usize>) -> Result<NDArray<T>, String> {

        let calculated_rank = shape.len(); 
        let mut calculated_size = 1; 
        for item in &shape {
            calculated_size *= item; 
        }

        Ok(Self {
            shape: Shape::new(shape),
            size: calculated_size,
            rank: calculated_rank,
            values: vec![T::default(); calculated_size],
        })
    }

    
    /// Create instance of NDArray, provide shape dimensions and array values as parameter
    pub fn array(shape: Vec<usize>, values: Vec<T>) -> Result<NDArray<T>, String> {

        let calculated_rank = shape.len(); 
        let mut calculated_size = 1; 
        for item in &shape {
            calculated_size *= item; 
        }

        if values.len() != calculated_size {
            return Err("Values don't match size based on dimensions".to_string()) 
        }

        Ok(Self {
            shape: Shape::new(shape),
            size: calculated_size,
            rank: calculated_rank,
            values: values,
        })
    }

    /// Fill ndarray with values
    pub fn fill(shape: Vec<usize>, value: T) -> Result<NDArray<T>, String> {
        let calculated_rank = shape.len(); 
        let mut calculated_size = 1; 
        for item in &shape {
            calculated_size *= item; 
        }

        let mut values = Vec::new(); 
        for _item in 0..calculated_size {
            values.push(value.clone());
        }


        Ok(Self {
            shape: Shape::new(shape),
            size: calculated_size,
            rank: calculated_rank,
            values: values,
        })
    }

    /// Reshape dimensions of array to new shape. Shape must match current size
    pub fn reshape(&mut self, shape_vals: Vec<usize>) -> Result<(), String> {

        if shape_vals.len() != self.rank {
            return Err("New Shape values don't match rank of array".to_string());
        }

        let mut size_validate = 1;
        for item in &shape_vals {
            size_validate *= item; 
        }

        if size_validate != self.size {
            return Err("New Shape values don't match size of array".to_string());
        }

        self.shape = Shape::new(shape_vals);
        Ok(())
    }

    /// Get contigous index of array using provided indices as parameter
    pub fn index(&self, indices: Vec<usize>) -> Result<usize, String> {

        if indices.len() != self.rank {
            return Err("Indexing doesn't match rank of ndarray".to_string());
        }

        let mut stride = 1; 
        let mut index = 0;
        let mut counter = self.rank;  
        for _n in 0..self.rank {
            let temp = stride * indices[counter-1]; 
            let curr_shape = self.shape.dim(counter-1);
            stride *= curr_shape;
            index += temp;  
            counter -= 1; 
        }

        if index > self.size-1 {
            return Err("Index out of bounds".to_string());
        }

        Ok(index)
    }

    /// Get indices from provided contigous index as parameter
    pub fn indices(&self, index: usize) -> Result<Vec<usize>, String> {

        if index > self.size-1 {
            return Err("Index out of bounds".to_string());
        }

        let mut indexs = vec![0; self.rank]; 
        let mut count = self.rank-1; 
        let mut curr_index = index; 
        for _n in 0..self.rank-1 {
            let dim_size = self.shape.dim(count);
            indexs[count] = curr_index % dim_size; 
            curr_index /= dim_size; 
            count -= 1;
        }

        indexs[0] = curr_index;
        Ok(indexs)       
    }

    /// Set index and generic value, index must be within size of array
    pub fn set_idx(&mut self, idx: usize, value: T) -> Result<(), String> {

        if idx > self.size {
            return Err("Index out of bounds".to_string());
        }

        self.values[idx] = value;
        Ok(())
    }

    /// Set generic value using provided indices. Indices must match rank of array
    pub fn set(&mut self, indices: Vec<usize>, value: T) -> Result<(), String> {

        if indices.len() != self.rank {
            return Err("Indices length don't match rank of ndarray".to_string());
        }

        let index = self.index(indices).unwrap();
        self.values[index] = value;
        Ok(())
    }


    /// Get rows dimension associated with multi dimensional array
    pub fn rows(&self, index: usize) -> Result<Vec<T>, String> {

        let dim_shape = self.shape.dim(0);
        let result_length = self.size() / dim_shape;
        let values = self.values();
        let mut start_index = index * result_length;
        let mut result = Vec::new();

        for _i in 0..result_length {
            let value = &values[start_index];
            result.push(value.clone());
            start_index += 1; 
        }
 
        Ok(result)

    }

    /// Get column dimension associated with multi dimensional array
    pub fn cols(&self, index: usize) -> Result<Vec<T>, String> {

        let mut result = Vec::new();
        let dim_shape = self.shape.dim(1);
        let values = self.values();
        let result_length = self.size() / dim_shape;
        let stride = dim_shape;
        let mut start = index; 

        for _i in 0..result_length {
            let value = &values[start];
            result.push(value.clone());
            start += stride; 
        }
 
        Ok(result)
    }

    /// Get values from a specific axis/slice
    pub fn axis(&self, axis: usize, index: usize) -> Result<NDArray<T>, String> {

        if axis > self.rank() - 1 { 
            return Err("Axis: Selected axis larger than rank".to_string());
        }

        if index > self.shape().dim(axis)-1 {
            return Err("Axis: Index for value is too large".to_string()); 
        }

        let mut values: Vec<T> = Vec::new();
        let mut new_shape = self.shape().clone();
        new_shape.remove(axis);
        let outer_size = new_shape.values().iter().product::<usize>();

        for item in 0..outer_size {
            let multi_index = new_shape.multi_index(item);
            let mut full_index = multi_index.clone();
            full_index.insert(axis, index); 
            let flat_index = self.index(full_index).unwrap();
            let val = &self.values()[flat_index];
            values.push(val.clone());
        }

        if new_shape.values().len() == 1 {
            new_shape.push(1);
        }
 
        Ok(NDArray::array(new_shape.values(),values).unwrap()) 
    }

    /// Get mutiple axis values with provided indices
    pub fn axis_indices(&self, axis: usize, indices: Vec<usize>) -> Result<NDArray<T>, String> {
 
        if axis > self.rank() - 1 { 
            return Err("Axis Indices: Selected axis larger than rank".to_string());
        }

        let mut feature_vec: Vec<T> = Vec::new();

        for idx in &indices {
            let axis_call = self.axis(axis, *idx).unwrap();
            let mut axis_values = axis_call.values().clone();
            feature_vec.append(&mut axis_values);
        }

        let mut shape = self.shape().values().clone();
        shape[axis] = indices.len();

        Ok(NDArray::array(shape, feature_vec).unwrap()) 

    }


    /// Drop specified axis of ndarray
    pub fn drop_axis(&self, axis: usize, index: usize) -> Result<NDArray<T>, String> {

        if axis > self.rank() - 1 { 
            let msg = "Drop Axis: Selected axis larger than rank";
            return Err(msg.to_string());
        }

        if index > self.shape().dim(axis) { 
            let msg = "Drop Axis: Selected indice too large for axis";
            return Err(msg.to_string());
        }

        if self.rank() > 2 {
            let msg = "Drop Axis: Only supported for rank 2 values";
            return Err(msg.to_string()); 
        }

        let mut shape_vals = self.shape().values();
        shape_vals[axis] -= 1;
        let mut result: NDArray<T> = NDArray::new(shape_vals).unwrap();

        let mut coords: Vec<usize> = vec![0, 0];
        let coord_len = coords.len() - 1;
        let axis_shape = self.shape().dim(axis); 
        for item in 0..axis_shape {
            let value = self.axis(axis, item).unwrap();
            if item != index {
                for val in value.values() {
                    result.set(coords.clone(), val.clone()).unwrap();
                    coords[coord_len - axis] += 1;
                }
                coords[coord_len - axis] = 0; 
                coords[axis] += 1;
            }
        }

        Ok(result)
    }
   

    /// Batch ndarray in specified amount of chunks of rows, cols etc.
    pub fn batch(&self, batch_size: usize) -> Result<Vec<NDArray<T>>, String> {
       
        if batch_size == 0 || batch_size >= self.size() {
            return Err("Batch size out of bounds".to_string())
        }

        if self.rank() != 2 {
            return Err("NDArray must be of rank 2".to_string())
        }

        let dim_size = batch_size * self.shape.dim(1);
        let mut start_index = 0; 
        let mut end_index = start_index + dim_size;

        let mut batches: Vec<NDArray<T>> = Vec::new();
        
        for _item in 0..self.size() {

            if end_index >= self.size()+1 {
                break;
            }

            let temp_vec: Vec<T> = self.values()[start_index..end_index].to_vec(); 
            let ndarray_batch: NDArray<T> = NDArray::array(
                vec![batch_size, self.shape.dim(1)], 
                temp_vec.clone()
            ).unwrap();

            batches.push(ndarray_batch); 
            start_index += self.shape.dim(1); 
            end_index += self.shape.dim(1); 
             
        }

        Ok(batches) 
    }


    pub fn value_indices(&self, value: T) -> Vec<usize> {
        self.values().iter()
            .enumerate()
            .filter_map(|(i, &ref x)| if *x == value { Some(i) } else { None })
            .collect()
    }


    pub fn indice_query(&self, indices: Vec<usize>) -> Result<NDArray<T>, String> {

        if indices.len() > self.size() {
            let msg = "Indices length is greater than array size";
            return Err(msg.to_string());
        }

        let mut values: Vec<T> = Vec::new();
        for idx in &indices {
        
            if *idx > self.size() {
                let msg = "Specified index greater than array size";
                return Err(msg.to_string()); 
            }

            let val = self.idx(*idx);
            values.push(val.clone());
        }

        Ok(NDArray::array(vec![values.len(), 1], values).unwrap())
    }

    pub fn split(
        &self, 
        axis: usize,
        percentage: f64) -> Result<(NDArray<T>, NDArray<T>), String> {

        if axis > self.shape().values().len() {
            let msg = "AXIS greater than current NDArray shape";
            return Err(msg.to_string());
        } 
        
        let axis_shape = self.shape().dim(axis);
        let split_dist = (percentage * axis_shape as f64).ceil();
        let rem = (axis_shape as f64 - split_dist).ceil();

        let mut x_shape: Vec<usize> = self.shape().values();
        let mut y_shape: Vec<usize> = self.shape().values();
        x_shape[axis] = split_dist as usize;
        y_shape[axis] = rem as usize;

        let mut x_vals: Vec<T> = Vec::new();
        let mut y_vals: Vec<T> = Vec::new();

        for axis_idx in 0..axis_shape {
            let item = self.axis(axis, axis_idx).unwrap();
            let mut x_item = item.values().clone();
            if axis_idx < split_dist as usize { 
                x_vals.append(&mut x_item);
            } else {
                y_vals.append(&mut x_item);
            }
        }

        let x: NDArray<T> = NDArray::array(x_shape, x_vals).unwrap();
        let y: NDArray<T> = NDArray::array(y_shape, y_vals).unwrap();
        Ok((x, y))
    }


}
