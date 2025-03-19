use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Sub}; 
use std::fmt::Debug; 
use std::cmp::PartialEq; 


/// Value node for computation graph
#[derive(Debug, Clone, Default)]
pub struct Tensor<T> {
    pub value: Rc<RefCell<T>>,
    pub gradient: Rc<RefCell<T>>,
}

impl<T: Clone> Tensor<T> {

    /// Create new instance of value (leafs/terminal nodes in graph)
    pub fn new(value: &T) -> Tensor<T> {
        
        Tensor {
            value: Rc::new(RefCell::new(value.clone())),
            gradient: Rc::new(RefCell::new(value.clone()))
        }
    }

    /// Create new instance of value with gradient
    pub fn full(value: &T, gradient: &T) -> Tensor<T> {
        
        Tensor {
            value: Rc::new(RefCell::new(value.clone())),
            gradient: Rc::new(RefCell::new(gradient.clone()))
        }
    }

    /// Get value associated with structure
    pub fn value(&self) -> T {
        self.value.borrow().clone()
    }

    /// Get gradient of value
    pub fn grad(&self) -> T {
        self.gradient.borrow().clone()
    }

    /// Set value associated with structure
    pub fn set_value(&self, val: &T) {
        self.value.replace(val.clone());
    }

    /// Set gradient of value
    pub fn set_grad(&self, grad: &T) {
        self.gradient.replace(grad.clone());
    }
}


#[macro_export]
macro_rules! tensor {

    ($val:expr) => {
        Tensor::new(&$val)
    };

    ($val:expr, $grad:expr) => {
        Tensor::full(&$val, &$grad)
    };

}










