use std::fmt::Debug; 
use crate::node::{Node}; 

/// Value node for computation graph
#[derive(Debug, Clone, Default)]
pub struct Tensor<T> {
    pub value: T,
    pub gradient: T,
}

impl<T: Clone> Tensor<T> {

    /// Create new instance of tensor value
    pub fn new(value: &T) -> Tensor<T> {
        
        Tensor {
            value: value.clone(),
            gradient: value.clone()
        }
    }

    /// Get value associated with structure
    pub fn value(&self) -> T {
        self.value.clone()
    }

    /// Get gradient of value
    pub fn grad(&self) -> T {
        self.gradient.clone()
    }
    /// Set value associated with structure
    pub fn set_value(&mut self, val: T) {
        self.value = val;
    }

    /// Set gradient of value
    pub fn set_grad(&mut self, grad: T) {
        self.gradient = grad;
    }
}

impl<T: Clone> Node<T> for Tensor<T> {

    fn forward(&mut self) -> T {
        self.value.clone()
    }

    fn backward(&mut self) {
        println!("Doing nothing"); 
    }

}


