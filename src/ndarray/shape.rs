use serde::{Serialize, Deserialize};


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Shape {
    values: Vec<usize>
}

impl Shape {

    /// Create instance of shape object for stride calculation
    pub fn new(shape_vals: Vec<usize>) -> Self {
        Self { values: shape_vals }
    }

    /// Get's the corresponding dimension of a shape vector
    pub fn dim(&self, index: usize) -> usize {
        self.values[index]
    }

    /// Get the vector values associated with shape object 
    pub fn values(&self) -> Vec<usize> {
        self.values.clone()
    }

    /// Reverse the shape indices
    pub fn reverse(&self) -> Vec<usize> {
        let mut cloned_shape = self.values.clone();
        cloned_shape.reverse();
        cloned_shape
    }

    /// Permute indices in shape vector
    pub fn permute(&self, indice_order: Vec<usize>) -> Vec<usize> {
        let mut new_shape: Vec<usize> = Vec::new();
        for item in &indice_order {
            new_shape.push(self.values[*item]);
        }
        new_shape
    }

    /// Produce 1d index from ndarray using higher rank index coordinates
    pub fn idx(&self, indices: Vec<usize>) -> usize { 
        let mut stride = 1; 
        let mut index = 0;
        let mut counter = indices.len();  
        for _n in 0..indices.len() {
            let temp = stride * indices[counter-1]; 
            let curr_shape = self.values[counter-1];
            stride *= curr_shape;
            index += temp;  
            counter -= 1; 
        }
        index
    }

    /// Produce multi index coordinate with 1d index supplied
    pub fn indices(&self, index: usize, rank: usize) -> Vec<usize> {
        let mut indexs = vec![0; rank]; 
        let mut count = rank-1; 
        let mut curr_index = index; 
        for _n in 0..rank-1 {
            let dim_size = self.values[count];
            indexs[count] = curr_index % dim_size; 
            curr_index /= dim_size; 
            count -= 1;
        }
        indexs[0] = curr_index;
        indexs
    }

    /// Get stride for provided axis (dimension)
    pub fn stride(&self, axis: usize) -> usize {
   
        let mut stride = 0; 
        let mut counter = self.values.len();
        let mut temp_stride = 1;

        for item in 0..self.values.len() { 
            let temp = temp_stride * self.values[counter-1];
            stride = temp/self.values[counter-1];
            temp_stride *= self.values[counter-1];
            counter -= 1;

            if item == axis {
                break;
            }
        }

        stride
    }

    /// Get stride for provided axis (dimension)
    pub fn axis_length(&self, axis: usize) -> usize {
  
        let mut shape_counter = 1;
        let mut temp_stride = 1;
        let mut remainder_length = self.values[axis];

        for item in 0..self.values.len() {

            if item == axis {
                break
            }

            let shape_temp = self.values[shape_counter];
            remainder_length = shape_temp * shape_counter;
            shape_counter += 1;
        }

        remainder_length

    }

}
