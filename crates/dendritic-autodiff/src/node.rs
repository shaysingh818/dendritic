use crate::tensor::Tensor; 
use std::fmt;
use std::cell::{RefCell, RefMut}; 


/// Base operation trait for allowing shared behavior for operations
pub trait Node<T> {

    /// method for defining forward pass behavior of operation
    fn forward(&mut self) -> T;

    /// method for defining backward pass behavior of operation
    fn backward(&mut self);
}


impl<T> fmt::Debug for Box<dyn Node<T>> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node trait") 
    }

}

/*
/// Base operation trait for allowing shared behavior for operations
pub trait Operation<T> {

    /// method for defining forward pass behavior of operation
    fn forward(
        &self, 
        inputs: Vec<Tensor<T>>, 
        prev: T) -> T;

    /// method for defining backward pass behavior of operation
    fn backward(
        &self, 
        inputs: &mut Vec<Tensor<T>>, 
        prev: &mut Node<T>,
        upstream_grad: T); 
}


impl<T> fmt::Debug for Box<dyn Operation<T>> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Operation trait") 
    }

}

/// Node structure that stores operations in a computation graph.
/// Nodes store the inputs and the outputs of the computed inputs. 
/// Nodes also store a trait object that contains shared behavior for all operations.
#[derive(Debug)]
pub struct Node<T> {
    pub inputs: Vec<Tensor<T>>,
    pub output: Tensor<T>,
    pub operation: Box<dyn Operation<T>>
}


impl<T: Clone> Node<T> {

    /// Construct binary node with 2 inputs
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

    /// Construct unary node with 1 inputs
    pub fn unary(
        rhs: T,
        op: Box<dyn Operation<T>>) -> Self {

        Node {
            inputs: vec![Tensor::new(&rhs)],
            output: Tensor::new(&rhs),
            operation: op
        }
    }

    /// Retrieve inputs stored in the node
    pub fn inputs(&self) -> &Vec<Tensor<T>> {
        &self.inputs
    }

    /// Retrieve output from computed inputs
    pub fn output(&self) -> &Tensor<T> {
        &self.output
    }

    /// Retrieves mutable reference of output tensor in node
    pub fn mut_output(&mut self) -> &mut Tensor<T> {
        &mut self.output
    }

    /// Retrieve output from computed inputs
    pub fn outputs(&self) -> Tensor<T> {
        self.output.clone()
    }

    /// Forward node that takes in optional reference to previous node
    pub fn forward(&mut self, input: Option<&Node<T>>) {

        if let Some(prev_node) = input {
            let output = self.operation.forward(
                self.inputs.clone(), 
                prev_node.output().value()
            );
            self.output = Tensor::new(&output); 
        } else {
            let output = self.operation.forward(
                self.inputs.clone(), 
                self.inputs[0].value()
            );
            self.output = Tensor::new(&output); 
        }

    }

    /// Perform backward pass for individual node instance
    pub fn backward(
        &mut self,
        prev: &mut Node<T>,
        upstream: Option<T>) {

        
        if let Some(upstream_grad) = upstream {
            self.operation.backward(
                &mut self.inputs, 
                prev,
                upstream_grad
            );
        } else {
            let temp_upstream = self.inputs[0].clone();
            self.operation.backward(
                &mut self.inputs,
                prev,
                temp_upstream.value()
            ); 
        }

    }

    /// Set value for specific index of input array
    pub fn set_input(&mut self, index: usize, value: T) {
        self.inputs[index].set_value(value); 
    }

    /// Set output of value attribute for node
    pub fn set_output(&mut self, index: usize, value: T) {
        self.output.set_value(value); 
    }


} */





