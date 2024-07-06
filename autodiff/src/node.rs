use ndarray::ndarray::NDArray;
use std::rc::Rc; 
use std::cell::{RefCell}; 


pub trait Node {
    fn forward(&mut self); 
    fn backward(&mut self, upstream_gradient: NDArray<f64>); 
    fn value(&self) -> NDArray<f64>;
    fn grad(&self) -> NDArray<f64>;
    fn set_grad(&mut self, upstream_gradient: NDArray<f64>);
}


#[derive(Debug, Clone, Default)]
pub struct Value<T> {
    pub value: Rc<RefCell<T>>,
    pub gradient: Rc<RefCell<T>>,
}

impl<T: Clone> Value<T> {

    pub fn new(value: &T) -> Value<T> {
        
        Value {
            value: Rc::new(RefCell::new(value.clone())),
            gradient: Rc::new(RefCell::new(value.clone()))
        }
    }

    pub fn val(&self) -> T {
        self.value.borrow().clone()
    }

    pub fn grad(&self) -> T {
        self.gradient.borrow().clone()
    }

    pub fn set_val(&mut self, val: &T) {
        self.value.replace(val.clone());
    }

    pub fn set_grad(&mut self, value: &T) {
        self.gradient.replace(value.clone());
    }

}


impl Node for Value<NDArray<f64>> {

    fn forward(&mut self) {} 

    fn set_grad(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient.replace(upstream_gradient);
    } 

    fn backward(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient.replace(upstream_gradient);        
    } 

    fn value(&self) -> NDArray<f64> { 
        self.value.borrow().clone()
    }

    fn grad(&self) -> NDArray<f64> { 
        self.gradient.borrow().clone()
    }

}
