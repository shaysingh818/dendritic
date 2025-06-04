use std::fmt; 
use crate::Local;
use std::fmt::{Debug, Display};
use crate::tensor::Tensor;
use crate::node::{Node}; 
use crate::graph::ComputationGraph; 
use ndarray::{arr2, Array2};


/// General purpose logging function
pub fn debug_log(msg: &str) {
    #[cfg(debug_assertions)]
    {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        println!("[{}]: {}", log_time, msg);
    }
}

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


#[derive(Clone, Debug)]
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


        debug_log(
            &format!(
                "Backward on raw value idx: {}",
                curr_idx
            ) 
        ); 

    }
}
