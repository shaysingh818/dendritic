use crate::tensor::Tensor; 
use crate::ops::Operation; 
use std::fmt;
use std::fmt::{Debug, Display}; 

/// Node structure that stores operations in a computation graph.
/// Nodes store the inputs and the outputs of the computed inputs. 
/// Nodes also store a trait object that contains shared behavior for all operations.
#[derive(Debug, Clone)]
pub struct Node<T> {
    pub inputs: Vec<usize>,
    pub upstream: Vec<usize>,
    pub value: Tensor<T>,
    pub operation: Operation<T>
}

impl<T: Clone + Default> Node<T> {

    /// Method to add input to operation
    pub fn inputs(&mut self) -> Vec<usize> {
        self.inputs.clone()
    }

    /// Method to add input to operation
    pub fn upstream(&self) -> Vec<usize> {
        self.upstream.clone()
    }

    /// Method to add input to operation
    pub fn add_input(&mut self, idx: usize) {
        self.inputs.push(idx)
    }

    /// Method to add input to operation
    pub fn add_upstream(&mut self, idx: usize) {
        self.upstream.push(idx)
    }

    /// Get output value of node structure
    pub fn output(&self) -> T {
        self.value.value.clone()
    }

    /// Set output attribute of node structure
    pub fn set_output(&mut self, val: T) {
        self.value = Tensor::new(&val);
    }

    /// Set gradient attribute of output tensor value
    pub fn set_grad_output(&mut self, val: T) {
        self.value.set_grad(val); 
    }

    /// Create value with no inputs (but contain upstream dependencies)
    pub fn val(value: T) -> Self {

        Node {
            inputs: vec![],
            upstream: vec![],
            value: Tensor::new(&value),
            operation: Operation::default(),
        }
    }

    /// Create binary node with exactly 2 inputs
    pub fn binary(lhs: usize, rhs: usize, op: Operation<T>) -> Self {

        Node {
            inputs: vec![lhs, rhs],
            upstream: vec![],
            value: Tensor::default(),
            operation: op,
        }
    }

    /// Perform forward pass on current node
    pub fn forward(
        &self, 
        nodes: &Vec<Node<T>>, 
        inputs: Vec<usize>,
        curr_node_idx: usize) -> T {
        (self.operation.forward)(nodes, inputs, curr_node_idx)
    }

    /// Peform backward pass on current node
    pub fn backward(
        &self, 
        nodes: &mut Vec<Node<T>>, 
        inputs: Vec<usize>,
        curr_node_idx: usize) {
        (self.operation.backward)(nodes, inputs, curr_node_idx)
    }

}


impl Node<f64> {

    /// Construct binary node with 2 inputs
    pub fn new() -> Self {

        Node {
            inputs: vec![],
            upstream: vec![],
            value: Tensor::default(),
            operation: Operation::default(),
        }
    }

}












