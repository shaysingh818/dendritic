use crate::tensor::Tensor; 
use crate::ops::Operation; 
use std::fmt;

// value trait that gets implemented for tensors and operations

// operation trait that works with value traits



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

    pub fn output(&self) -> T {
        self.value.value.clone()
    }

    pub fn set_output(&mut self, val: T) {
        self.value = Tensor::new(&val);
    }

    pub fn forward(
        &self, 
        nodes: &Vec<Node<T>>, 
        inputs: Vec<usize>) -> T {
        (self.operation.forward)(nodes, inputs)
    }

}

impl Node<f64> {

    /// Construct binary node with 2 inputs
    pub fn new() -> Self {

        Node {
            inputs: vec![],
            upstream: vec![],
            value: Tensor::default(),
            operation: Operation::add(),
        }
    }

    /// Construct binary node with 2 inputs
    pub fn val(value: f64) -> Self {

        Node {
            inputs: vec![],
            upstream: vec![],
            value: Tensor::new(&value),
            operation: Operation::add(),
        }
    }

    /// Construct binary node with 2 inputs
    pub fn binary(lhs: usize, rhs: usize, op: Operation<f64>) -> Self {

        Node {
            inputs: vec![lhs, rhs],
            upstream: vec![],
            value: Tensor::default(),
            operation: op,
        }
    }



    /*
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
    pub fn output(&self) -> Tensor<T> {
        self.output.clone()
    }

    /// Retrieves mutable reference of output tensor in node
    pub fn mut_output(&mut self) -> &mut Tensor<T> {
        &mut self.output
    }

    /// Retrieve output from computed inputs
    pub fn outputs(&self) -> Tensor<T> {
        self.output.clone()
    }

    /// Set value for specific index of input array
    pub fn set_input(&mut self, index: usize, value: T) {
        self.inputs[index].set_value(value); 
    }

    /// Set output of value attribute for node
    pub fn set_output(&mut self, value: T) {
        self.output.set_value(value); 
    }

    /// Set output of value attribute for node
    pub fn set_grad_output(&mut self, value: T) {
        self.output.set_grad(value); 
    }

    /// Forward node that takes in optional reference to previous node
    pub fn forward(&mut self, input: Option<&Node<T>>) {

        if let Some(prev_node) = input {
            let output = self.operation.forward(
                self.inputs.clone(), 
                prev_node.output()
            );
            self.output = Tensor::new(&output); 
        } else {
            let output = self.operation.forward(
                self.inputs.clone(), 
                self.inputs[0].clone()
            );
            self.output = Tensor::new(&output); 
        }

    }

    /// Perform backward pass for individual node instance
    pub fn backward(&mut self, prev_node: &mut Node<T>) {
        self.operation.backward(&mut self.inputs, prev_node);
    }
    */

}












