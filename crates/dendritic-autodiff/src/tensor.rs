use std::cell::RefCell;
use std::rc::Rc;
use std::fmt::Debug; 
use std::cmp::PartialEq; 

/// Value node for computation graph
#[derive(Debug, Clone, Default)]
pub struct Tensor<T> {
    pub value: T,
    pub gradient: T,
}

impl<T: Clone> Tensor<T> {

    /// Create new instance of value (leafs/terminal nodes in graph)
    pub fn new(value: &T) -> Tensor<T> {
        
        Tensor {
            value: value.clone(),
            gradient: value.clone()
        }
    }

    /// Get value associated with structure
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get gradient of value
    pub fn grad(&self) -> &T {
        &self.gradient
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

