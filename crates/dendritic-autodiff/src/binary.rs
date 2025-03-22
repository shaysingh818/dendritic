use crate::graph::Dendrite;
use crate::node::Node; 
use crate::ops::{Add, Sub}; 


/// Binary operations for types of f64
pub trait BinaryOperation {

    fn add(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64>; 

    fn sub(&mut self, lhs: f64, rhs: f64) -> &mut Dendrite<f64>; 

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
