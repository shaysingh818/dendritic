use std::fmt::Debug;
use std::collections::{HashMap, HashSet}; 

use crate::operations::base::*; 

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

}


