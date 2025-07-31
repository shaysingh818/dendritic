use std::fmt; 
use std::fmt::Debug;

use serde::{Serialize, Deserialize}; 
use log::debug; 

use crate::node::{Node}; 


pub trait Operation<T>: OperationClone<T> + Debug {

    fn forward(&self, nodes: &Vec<Node<T>>, curr_idx: usize) -> T;
 
    fn backward(&self, nodes: &mut Vec<Node<T>>, curr_idx: usize);

}

pub trait OperationClone<T> {
    fn clone_box(&self) -> Box<dyn Operation<T>>;
}

impl<T, U> OperationClone<T> for U
where
    U: 'static + Operation<T> + Clone,
{
    fn clone_box(&self) -> Box<dyn Operation<T>> {
        Box::new(self.clone())
    }
}


impl<T> Clone for Box<dyn Operation<T>> {
    fn clone(&self) -> Box<dyn Operation<T>> {
        self.clone_box()
    }
}

impl<T> fmt::Display for dyn Operation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trait: {}", self)
    }
} 


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefaultValue; 


impl<T> Operation<T> for DefaultValue 
where 
    T: Clone + Default, 
{

    fn forward(
        &self, 
        nodes: &Vec<Node<T>>, 
        curr_idx: usize) -> T {

        nodes[curr_idx].output()
    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<T>>, 
        curr_idx: usize) {


        debug!(
            "Backward on raw value idx: {}",
            curr_idx
        ); 

    }
}

