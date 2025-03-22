use crate::tensor::Tensor; 
use std::fmt; 

pub trait Operation<T> {

    fn forward(&self, inputs: &mut Vec<Tensor<T>>) -> T;

    fn backward(&self, inputs: &mut Vec<Tensor<T>>); 
}


impl<T> fmt::Debug for Box<dyn Operation<T>> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Operation trait") 
    }

}

/// Base node structure that captures an operation
#[derive(Debug)]
pub struct Node<T> {
    pub inputs: Vec<Tensor<T>>,
    pub output: Tensor<T>,
    pub operation: Box<dyn Operation<T>>
}


impl<T: Clone> Node<T> {

    pub fn binary(
        lhs: T, 
        rhs: T,
        op: Box<dyn Operation<T>>) -> Self {

        Node {
            inputs: vec![
                Tensor::new(&lhs),
                Tensor::new(&rhs)
            ],
            output: Tensor::new(&rhs),
            operation: op
        }
    }

    pub fn inputs(&self) -> &Vec<Tensor<T>> {
        &self.inputs
    }

    pub fn output(&self) -> &Tensor<T> {
        &self.output
    }

    pub fn forward(&mut self) {
        let output = self.operation.forward(&mut self.inputs);
        self.output = Tensor::new(&output); 
    }

    pub fn backward(&mut self) {
        self.operation.backward(&mut self.inputs);
    }

}

