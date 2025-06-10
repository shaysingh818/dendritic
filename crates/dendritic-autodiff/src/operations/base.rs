use std::fmt; 
use std::fmt::{Debug, Display};

use serde::{Serialize, Serializer, Deserialize}; 
use chrono::Local;
use ndarray::{arr2, Array2};
use log::{debug, warn, info}; 

use crate::tensor::Tensor;
use crate::node::{Node}; 
use crate::graph::ComputationGraph; 


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
        nodes: &mut Vec<Node<T>>, 
        curr_idx: usize) {


        debug!(
            "Backward on raw value idx: {}",
            curr_idx
        ); 

    }
}
