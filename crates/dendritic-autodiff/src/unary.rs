use crate::graph::Dendrite;
use crate::node::Node;
use crate::ops::{Add}; 


/// Unary operations for scalar values
pub trait UnaryOperation {

    fn u_add(&mut self, rhs: f64) -> &mut Dendrite<f64>; 

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

}
