use std::cell::RefCell;
use std::rc::Rc;
use std::fmt::Debug;
use std::fmt; 
use std::cmp::PartialEq; 
use crate::tensor::Tensor; 


/// Base node structure that captures an operation
#[derive(Debug, Clone)]
pub struct Node<T> {
    pub inputs: Vec<Tensor<T>>,
    pub output: Tensor<T>,
    pub forward: fn(node: &mut Node<T>),
    pub backward: fn(node: &Node<T>),
}


#[derive(Clone)]
pub struct Node2<T> {
    pub inputs: Vec<Tensor<T>>,
    pub output: Tensor<T>,
    pub operation: Box<dyn Operation2<T>>
}

impl<T: Clone> Node<T> {

    pub fn binary(
        lhs: T, 
        rhs: T,
        forward: fn(node: &mut Node<T>),
        backward: fn(node: &Node<T>)) -> Self {

        Node {
            inputs: vec![
                Tensor::new(&lhs),
                Tensor::new(&rhs)
            ],
            output: Tensor::new(&rhs),
            forward: forward,
            backward: backward

        }
    }

    pub fn inputs(&self) -> &Vec<Tensor<T>> {
        &self.inputs
    }

    pub fn output(&self) -> &Tensor<T> {
        &self.output
    }

    pub fn forward(&mut self) {
        (self.forward)(self)
    }

    pub fn backward(&mut self) {
        (self.backward)(self)
    }

}


impl<T: Clone> Node2<T> {

    pub fn new(lhs: T, rhs: T, op: Box<dyn Operation2<T>>) -> Self {

        Node2 {
            inputs: vec![
                Tensor::new(&lhs),
                Tensor::new(&rhs)
            ],
            output: Tensor::new(&rhs),
            operation: op
        }
    }

    pub fn forward(&mut self) {
        self.operation.forward(self)
    }

    /*
    pub fn backward(&mut self) {
        self.operation.backward(self)
    } */

}


/// Computation Graph Structure (genrically defined)
#[derive(Debug, Clone)]
pub struct Dendrite<T> {
    pub nodes: Vec<Node<T>>,
    pub current_node: usize,
    pub reference_count: usize,
}

impl<T: Clone> Dendrite<T> {

    pub fn new() -> Self {

        Dendrite {
            nodes: vec![],
            current_node: 0,
            reference_count: 0
        }
    }

    pub fn reference_count(&self) -> usize {
        self.reference_count
    }

    pub fn current_node_idx(&self) -> usize {
        self.current_node
    }

    pub fn current_node(&self) -> &Node<T> {
        &self.nodes[self.current_node]
    }

    pub fn nodes(&self) -> &Vec<Node<T>> {
        &self.nodes
    }

}


pub trait Operation<T> {

    fn forward(node: &mut Node<T>);

    fn backward(node: &Node<T>); 
}


pub trait Operation2<T> {

    fn forward(&self, node: &mut Node2<T>);

    fn backward(&self, node: &Node2<T>); 
}


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
            Add::forward,
            Add::backward
        );

        self.nodes.push(node);
        self
    }

    fn sub(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64> {

        let node = Node::binary(
            lhs, 
            rhs, 
            Sub::forward, 
            Sub::backward
        );

        self.nodes.push(node);
        self
    }
}

impl UnaryOperation for Dendrite<f64> {

    fn u_add(&mut self, rhs: f64) -> &mut Dendrite<f64> {

        let lhs = self.nodes[self.current_node].output.clone();
        let node = Node::binary(
            lhs.value, 
            rhs,
            Add::forward,
            Add::backward
        );

        self.nodes.push(node);
        self
        
    }

}


#[derive(Debug, Clone)]
pub struct Add;

#[derive(Debug, Clone)]
pub struct Add2;


impl Operation2<f64> for Add2 {

    fn forward(&self, mut node: &mut Node2<f64>) {
        //println!("Forward addition");
        let lhs = &node.inputs[0].value; 
        let rhs = &node.inputs[1].value;
        let output = *lhs + *rhs;
        node.output = Tensor::new(&output); 
    }

    fn backward(&self, node: &Node2<f64>) {
        //println!("Backward addition"); 
    }

}

impl Operation<f64> for Add {

    fn forward(mut node: &mut Node<f64>) {
        //println!("Forward addition");
        let lhs = &node.inputs[0].value; 
        let rhs = &node.inputs[1].value;
        let output = *lhs + *rhs;
        node.output = Tensor::new(&output); 
    }

    fn backward(node: &Node<f64>) {
        //println!("Backward addition"); 
    }

}

#[derive(Debug, Clone)]
pub struct Sub; 

impl Operation<f64> for Sub {

    fn forward(mut node: &mut Node<f64>) {
        //println!("Forward subtraction");
        let lhs = &node.inputs[0].value; 
        let rhs = &node.inputs[1].value;
        let output = *lhs - *rhs;
        node.output = Tensor::new(&output); 
    }


    fn backward(node: &Node<f64>) {
        //println!("Backward subtraction"); 
    }

}
