use std::io::Error; 
use std::collections::HashMap;

use ndarray::Array2;
use serde::{Serialize, Deserialize}; 

use crate::autodiff::tensor::Tensor; 
use crate::autodiff::operations::base::*; 

/// Node structure that stores operations in a computation graph.
/// Nodes store the inputs and the outputs of the computed inputs. 
/// Nodes also store a trait object that contains shared behavior for all operations.
#[derive(Debug, Clone)]
pub struct Node<T> {

    /// Indicates if node value is a parameter
    pub is_param: bool,

    /// Graph node indices  associated with inputs to node
    pub inputs: Vec<usize>,

    /// Graph node indices associated with upstream values
    pub upstream: Vec<usize>,

    /// Generic tensor to store value in node
    pub value: Tensor<T>,

    /// Generic trait for operation behavior of nodes
    pub operation: Box<dyn Operation<T>>,
}

/// Serializable struct for node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSerialize<T> {
    pub is_param: bool,
    pub inputs: Vec<usize>,
    pub upstream: Vec<usize>,
    pub value: Tensor<T>,
    pub operation: String
}

/// Trait for serializing and deserializing node structure
pub trait NodeSerialization<T> {

    /// Trait method to save node and get prettified json string
    fn save(&self) -> Result<String, Error>; 

    /// Trait method to load node instance
    fn load(
        node: NodeSerialize<T>, 
        op_registry: HashMap<String, Box<dyn Operation<T>>>) -> std::io::Result<Node<T>>;

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

    /// Get operation trait associated with node
    pub fn operation(&self) -> Box<dyn Operation<T>> {
        self.operation.clone()
    }

    /// Set output attribute of node structure
    pub fn set_output(&mut self, val: T) {
        self.value.set_value(val);
    }

    /// Set gradient attribute of output tensor value
    pub fn set_grad_output(&mut self, val: T) {
        self.value.set_grad(val); 
    }

    /// Set operation for specific node shared behavior
    pub fn set_operation(&mut self, op: Box<dyn Operation<T>>) {
        self.operation = op; 
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
            is_param: false,
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
            is_param: false,
            inputs: vec![lhs, rhs],
            upstream: vec![],
            value: Tensor::default(),
            operation: op,
        }
    }

    /// Create unary node with exactly 1 inputs
    pub fn unary(
        lhs: usize, 
        op: Box<dyn Operation<T>>) -> Self {

        Node {
            is_param: false,
            inputs: vec![lhs],
            upstream: vec![],
            value: Tensor::default(),
            operation: op,
        }
    }

    /// Create serializable version of structure
    pub fn serialize(&self) -> NodeSerialize<T> {

        NodeSerialize {
            is_param: self.is_param,
            inputs: self.inputs.clone(),
            upstream: self.upstream.clone(),
            value: self.value.clone(),
            operation: format!("{:?}", self.operation)
        }

    }

}

macro_rules! node_serialize {

    ($t:ty) => {

        impl NodeSerialization<$t> for Node<$t> {

            fn save(&self) -> Result<String, Error> {

                let obj = NodeSerialize {
                    is_param: self.is_param,
                    inputs: self.inputs.clone(),
                    upstream: self.upstream.clone(),
                    value: self.value.clone(),
                    operation: format!("{:?}", self.operation)
                };

                Ok(serde_json::to_string_pretty(&obj).unwrap())
            }


            /// Convert to structure that is serializable
            fn load(
                node: NodeSerialize<$t>, 
                op_registry: HashMap<String, Box<dyn Operation<$t>>>) -> std::io::Result<Node<$t>> {

                let key = node.operation.to_string();
                
                match op_registry.get(&key) {
                    Some(op) => {

                        Ok(Node {
                            is_param: node.is_param,
                            inputs: node.inputs,
                            upstream: node.upstream,
                            value: node.value,
                            operation: op.clone(),
                        })
                    },
                    _ => panic!("Couldn't find matching behavior trait for {key}")
                }

            }

        }

    }

}

node_serialize!(f64); 
node_serialize!(Array2<f64>);


#[cfg(test)]
mod node_tests {

    use crate::autodiff::node::{Node};
    use crate::autodiff::operations::arithmetic::*;
    

    #[test]
    fn test_node_constructors() {

        let node_val = Node::val(10.0); 
        assert_eq!(node_val.inputs().len(), 0); 
        assert_eq!(node_val.upstream().len(), 0); 
        assert_eq!(node_val.output(), 10.0);

        let binary_node_val: Node<f64> = Node::binary(
            0, 1, 
            Box::new(Add)
        );

        assert_eq!(binary_node_val.inputs().len(), 2); 
        assert_eq!(binary_node_val.inputs(), vec![0, 1]);
        assert_eq!(binary_node_val.upstream().len(), 0);

    }

    #[test]
    fn test_node_forward() {

        let mut nodes: Vec<Node<f64>> = Vec::new();

        // these go out of scope after adding to node vector
        let a: Node<f64> = Node::val(5.0); 
        let b: Node<f64> = Node::val(10.0);
        let add_node = Node::binary(0, 1, Box::new(Add));

        nodes.push(a); 
        nodes.push(b); 
        nodes.push(add_node.clone()); 

        let a_output = nodes[0].forward(&nodes, 0); 
        let b_output = nodes[1].forward(&nodes, 1); 
        let output = nodes[2].forward(&nodes, 2); 

        assert_eq!(a_output, 5.0); 
        assert_eq!(b_output, 10.0); 
        assert_eq!(output, 15.0); 
    }

    #[test]
    fn test_node_backward() {

        let mut nodes: Vec<Node<f64>> = Vec::new();

        // these go out of scope after adding to node vector
        let a: Node<f64> = Node::val(5.0); 
        let b: Node<f64> = Node::val(10.0);
        let add_node = Node::binary(0, 1, Box::new(Add));

        nodes.push(a); 
        nodes.push(b); 
        nodes.push(add_node.clone()); 

        let output = nodes[2].forward(&nodes, 2);

        let mut node_2 = nodes[2].clone(); 
        node_2.set_output(output); 
        node_2.backward(&mut nodes, 2);

        assert_eq!(nodes[1].grad(), 1.0); 
        assert_eq!(nodes[0].grad(), 1.0); 
    }

}
