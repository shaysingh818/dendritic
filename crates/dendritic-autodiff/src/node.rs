use dendritic_ndarray::ndarray::NDArray;
use std::rc::Rc; 
use std::cell::{RefCell}; 


/// Methods for each value in computation graph
pub trait Node {
    fn forward(&mut self); 
    fn backward(&mut self, upstream_gradient: NDArray<f64>); 
    fn value(&self) -> NDArray<f64>;
    fn grad(&self) -> NDArray<f64>;
    fn set_grad(&mut self, upstream_gradient: NDArray<f64>);
}


/// Value node for computation graph
#[derive(Debug, Clone, Default)]
pub struct Value<T> {
    pub value: Rc<RefCell<T>>,
    pub gradient: Rc<RefCell<T>>,
}

impl<T: Clone> Value<T> {

    /// Create new instance of value for comptuation graph
    pub fn new(value: &T) -> Value<T> {
        
        Value {
            value: Rc::new(RefCell::new(value.clone())),
            gradient: Rc::new(RefCell::new(value.clone()))
        }
    }

    /// Get value associated with structure
    pub fn val(&self) -> T {
        self.value.borrow().clone()
    }

    /// Get gradient of value
    pub fn grad(&self) -> T {
        self.gradient.borrow().clone()
    }

    /// Set value associated with structure
    pub fn set_val(&mut self, val: &T) {
        self.value.replace(val.clone());
    }

    /// Set gradient of value in computation graph
    pub fn set_grad(&mut self, value: &T) {
        self.gradient.replace(value.clone());
    }

}


impl Node for Value<NDArray<f64>> {

    /// Forward operation for a value
    fn forward(&mut self) {} 

    /// Set gradient from upstream for value
    fn set_grad(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient.replace(upstream_gradient);
    } 

    /// Set gradient from upstream in backward pass
    fn backward(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient.replace(upstream_gradient);        
    } 

    /// Retrieve value from node in computation graph
    fn value(&self) -> NDArray<f64> { 
        self.value.borrow().clone()
    }

    /// Retrieve gradient from node in computation graph
    fn grad(&self) -> NDArray<f64> { 
        self.gradient.borrow().clone()
    }

}
