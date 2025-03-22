use std::fmt::Debug;
use std::fmt; 
use std::any::type_name;
use crate::tensor::Tensor;
use crate::node::Node; 


pub fn log_type<T>(value: T) {
    println!("Type of value is: {}", type_name::<T>());
}


/*
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

#[derive(Debug, Clone)]
pub struct Add;

impl Operation<f64> for Add {

    fn forward(&self, inputs: &mut Vec<Tensor<f64>>) -> f64 {
        let lhs = inputs[0].value; 
        let rhs = inputs[1].value;
        lhs + rhs
    }

    fn backward(&self, inputs: &mut Vec<Tensor<f64>>) {
        //println!("Backward addition"); 
    }

}

#[derive(Debug, Clone)]
pub struct Sub; 

impl Operation<f64> for Sub {

    fn forward(&self, inputs: &mut Vec<Tensor<f64>>) -> f64 {
        let lhs = inputs[0].value; 
        let rhs = inputs[1].value;
        lhs - rhs
    }


    fn backward(&self, node: &mut Vec<Tensor<f64>>) {
        //println!("Backward subtraction"); 
    }

}

*/

/// Computation Graph Structure (genrically defined)
pub struct Dendrite<T> {
    pub nodes: Vec<Node<T>>,
    pub current_node_idx: usize,
    pub reference_count: usize,
}

impl<T: Clone> Dendrite<T> {

    pub fn new() -> Self {

        Dendrite {
            nodes: vec![],
            current_node_idx: 0,
            reference_count: 0
        }
    }

    pub fn reference_count(&self) -> usize {
        self.reference_count
    }

    pub fn current_node_idx(&self) -> usize {
        self.current_node_idx
    }

    pub fn current_node(&self) -> &Node<T> {
        &self.nodes[self.current_node_idx]
    }

    pub fn nodes(&self) -> &Vec<Node<T>> {
        &self.nodes
    }

    pub fn forward(&mut self) {
        for mut node in &mut self.nodes {
            node.forward();
        }
    }

}


/*
/// Binary operations for types of f64
pub trait BinaryOperation {

    fn add(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64>; 

    fn sub(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64>; 

}


/// Unary operations for scalar values
pub trait UnaryOperation {

    fn u_add(&mut self, rhs: f64) -> &mut Dendrite<f64>; 

}

impl BinaryOperation for Dendrite<f64> {

    fn add(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64> {

        let node = Node::binary(
            lhs, 
            rhs,
            Box::new(Add)
        );

        self.nodes.push(node);
        self
    }

    fn sub(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64> {

        let node = Node::binary(
            lhs, 
            rhs,
            Box::new(Sub)
        );

        self.nodes.push(node);
        self
    }
}

impl UnaryOperation for Dendrite<f64> {

    fn u_add(&mut self, rhs: f64) -> &mut Dendrite<f64> {

        let lhs = self.nodes[self.current_node_idx].output.clone();
        let node = Node::binary(
            lhs.value, 
            rhs,
            Box::new(Add)
        );

        self.nodes.push(node);
        self
        
    }

} */ 


