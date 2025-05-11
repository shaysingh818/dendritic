use crate::tensor::Tensor; 
use crate::ops::*; 
use std::fmt;
use std::fmt::{Debug, Display}; 
use std::cell::RefCell;
use std::borrow::{BorrowMut, Borrow};

/// Node structure that stores operations in a computation graph.
/// Nodes store the inputs and the outputs of the computed inputs. 
/// Nodes also store a trait object that contains shared behavior for all operations.
#[derive(Debug, Clone)]
pub struct Node<T> {
    pub inputs: Vec<usize>,
    pub upstream: Vec<usize>,
    pub value: Tensor<T>,
    pub operation: Box<dyn Operation<T>>,
}

impl<T: Clone + Default> Node<T> {

    /// Method to add input to operation
    pub fn inputs(&self) -> Vec<usize> {
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
        self.value.value()
    }

    /// Get output value of node structure
    pub fn grad(&self) -> T {
        self.value.grad().clone()
    }

    /// Set output attribute of node structure
    pub fn set_output(&mut self, val: T) {
        self.value.set_value(val);
    }

    /// Set gradient attribute of output tensor value
    pub fn set_grad_output(&mut self, val: T) {
        self.value.set_grad(val); 
    }

    /// Perform forward pass on current node
    pub fn forward(&self, nodes: &Vec<Node<T>>, curr_node_idx: usize) -> T {
        self.operation.forward(nodes, curr_node_idx)
    }

    /// Peform backward pass on current node
    pub fn backward(&mut self, nodes: &mut Vec<Node<T>>, curr_node_idx: usize) {
        self.operation.backward(nodes, curr_node_idx)
    }

    /// Create value with no inputs (but contain upstream dependencies)
    pub fn val(value: T) -> Self {

        let val = Box::new(DefaultValue);

        Node {
            inputs: vec![],
            upstream: vec![],
            value: Tensor::new(&value),
            operation: val,
        }
    }

    /// Create binary node with exactly 2 inputs
    pub fn binary(
        lhs: usize, 
        rhs: usize, 
        op: Box<dyn Operation<T>>) -> Self {

        Node {
            inputs: vec![lhs, rhs],
            upstream: vec![],
            value: Tensor::default(),
            operation: op,
        }
    }

}



/*
#[derive(Debug, Clone)]
pub struct Node2<T> {
    pub inputs: Vec<usize>,
    pub upstream: Vec<usize>,
    pub value: Tensor<T>,
    pub operation: Box<dyn OperationTrait<T>>,
}


impl<T> Node2<T> 
where
    T: Clone + Default,
{

    /// Method to add input to operation
    pub fn inputs(&self) -> Vec<usize> {
        self.inputs.clone()
    }

    /// Get output value of node structure
    pub fn output(&self) -> T {
        self.value.value.clone()
    }

    /// Get output value of node structure
    pub fn grad(&self) -> T {
        self.value.grad().clone()
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

    /// Set output attribute of node structure
    pub fn set_output(&mut self, val: T) {
        self.value = Tensor::new(&val);
    }

    /// Set gradient attribute of output tensor value
    pub fn set_grad_output(&mut self, val: T) {
        self.value.set_grad(val); 
    }

    /// Perform forward pass on current node
    pub fn forward(
        &self, 
        nodes: &Vec<Node2<T>>, 
        curr_node_idx: usize) -> T {

        self.operation.forward(nodes, curr_node_idx)
    }

    /// Peform backward pass on current node
    pub fn backward(
        &self, 
        nodes: &mut Vec<Node2<T>>, 
        curr_node_idx: usize) {

        self.operation.backward(nodes, curr_node_idx)
    }

    /// Create a value node with no inputs and a default operation
    pub fn val(value: T) -> Self {

        let val = Box::new(DefaultValue);

        Node2 {
            inputs: vec![],
            upstream: vec![],
            value: Tensor::new(&value),
            operation: val,
        }
    }

    /// Create binary node with exactly 2 inputs
    pub fn binary(
        lhs: usize, 
        rhs: usize, 
        op: Box<dyn OperationTrait<T>>) -> Self {

        Node2 {
            inputs: vec![lhs, rhs],
            upstream: vec![],
            value: Tensor::default(),
            operation: op,
        }
    }

} */









