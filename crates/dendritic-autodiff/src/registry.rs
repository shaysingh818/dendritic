use std::fmt::Debug;
use std::collections::{HashMap, HashSet}; 

use crate::operations::base::*; 
use crate::operations::activation::*; 
use crate::operations::arithmetic::*; 
use crate::operations::loss::*; 

pub struct OperationRegistry<T> {
    registry: HashMap<String, Box<dyn Operation<T>>>
}

impl<T: Clone + Default + Debug> OperationRegistry<T> {

    pub fn new() -> Self {
        OperationRegistry {
            registry: HashMap::new()
        }
    }

    pub fn register(&mut self, key: &str, op: Box<dyn Operation<T>>) {
        self.registry.insert(key.to_string(), op);
    }

    pub fn arithmetic(&mut self) {
        self.register("Add", Box::new(Add)); 
        self.register("Mul", Box::new(Mul)); 
        self.register("Sub", Box::new(Sub)); 
    }
 
    pub fn activation(&mut self) {
        self.register("Tanh", Box::new(Tanh)); 
        self.register("Sigmoid", Box::new(Sigmoid)); 
    }

    pub fn loss(&mut self) {
        self.register("MSE", Box::new(MSE)); 
        self.register("BinaryCrossEntropy", Box::new(BinaryCrossEntropy));
        self.register("DefaultLossFunction", Box::new(BinaryCrossEntropy)); 
    }

}


