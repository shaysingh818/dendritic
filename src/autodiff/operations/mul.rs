use std::fmt; 
use std::fmt::Debug;

use serde::{Serialize, Deserialize}; 
use ndarray::{Array2, Axis};
use log::debug; 

use crate::autodiff::operations::base::*; 
use crate::autodiff::node::{Node}; 
use crate::autodiff::graph::ComputationGraph; 


pub trait TestOps<T> {

    fn mul2(&mut self, inputs: Vec<T>) -> &mut ComputationGraph<T>; 

    fn pow(&mut self, inputs: Vec<T>) -> &mut ComputationGraph<T>;

}

impl TestOps<f64> for ComputationGraph<f64> {

    fn mul2(&mut self, inputs: Vec<f64>) -> &mut ComputationGraph<f64> {
        
        match inputs.len() {
            2 => {
                self.binary(Some(inputs[0].clone()), Some(inputs[1].clone()), Box::new(MUL))
            },

            1 => {
                self.unary(inputs[0].clone(), Box::new(MUL))
            },
            _ => {
                panic!("add: One or 2 inputs required");
            }
        }
    }


    fn pow(&mut self, inputs: Vec<f64>) -> &mut ComputationGraph<f64> {
        
        match inputs.len() {
            2 => {
                self.binary(Some(inputs[0].clone()), Some(inputs[1].clone()), Box::new(POW))
            },

            1 => {
                self.unary(inputs[0].clone(), Box::new(POW))
            },
            _ => {
                panic!("add: One or 2 inputs required");
            }
        }
    }

}



#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MUL;


impl fmt::Display for MUL {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trait: {}", self)
    }
} 

impl Operation<f64> for MUL {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug!(
            "(MUL) Performing forward pass on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() * nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {


        debug!(
            "(MUL) Performing backward pass on node index: {:?}",
            curr_idx
        );

        let node_inputs = nodes[curr_idx].inputs();
        let lhs = nodes[node_inputs[0]].output(); 
        let rhs = nodes[node_inputs[1]].output();

        nodes[node_inputs[0]].set_grad_output(rhs); 
        nodes[node_inputs[1]].set_grad_output(lhs);

        debug!(
            "(MUL) Updated gradients for node input indexes: {:?}",
            node_inputs
        ); 

    }
}


impl Operation<Array2<f64>> for MUL {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            "(MUL) Performing forward pass on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();
        debug!(
            "[MUL]: [Node {:?}]: {:?} * [Node {:?}]: {:?}", 
            inputs[0], lhs.dim(), inputs[1], rhs.dim()
        ); 
        lhs.dot(&rhs)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        debug!(
            "(MUL) Performing backward multiply on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        let upstream = nodes[curr_idx].upstream();

        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();

        if upstream.len() > 1 {
            panic!("Backward multiply can only handle one upstream"); 
        }

        debug!("[MUL]: Upstream node {:?}", upstream[0]); 

        match upstream.len() {
            1 => {
                let upstream = nodes[upstream[0]].grad();
                let rhs_grad = upstream.dot(&rhs.t());
                let lhs_grad = lhs.t().dot(&upstream);

                nodes[inputs[0]].set_grad_output(rhs_grad); 
                nodes[inputs[1]].set_grad_output(lhs_grad);
            },
            0 => {
                panic!("No upstream values associated with node: {:?}", nodes[curr_idx]); 
            },
            _ => {
                panic!("MUL: Unable to handle and map upstream values for backward pass");
            }
        }

        debug!(
            "(MUL) Updated gradients for node input indexes: {:?}",
            inputs
        ); 

    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct POW;


impl fmt::Display for POW {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trait: {}", self)
    }
} 

impl Operation<f64> for POW {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug!(
            "(MUL) Performing forward pass on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() * nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {


        debug!(
            "(MUL) Performing backward pass on node index: {:?}",
            curr_idx
        );

        let node_inputs = nodes[curr_idx].inputs();
        let lhs = nodes[node_inputs[0]].output(); 
        let rhs = nodes[node_inputs[1]].output();

        nodes[node_inputs[0]].set_grad_output(rhs); 
        nodes[node_inputs[1]].set_grad_output(lhs);

        debug!(
            "(MUL) Updated gradients for node input indexes: {:?}",
            node_inputs
        ); 

    }
}



#[cfg(test)]
mod mul_ops_test {

    use crate::autodiff::graph::*;
    use crate::autodiff::operations::arithmetic::*; 
    use crate::autodiff::operations::loss::*; 
    use crate::autodiff::operations::mul::*; 
    use ndarray::{arr2};


    #[test]
    fn test_multiply() {

        let mut scalar_graph: ComputationGraph<f64> = ComputationGraph::new();
        scalar_graph.binary(Some(3.0), Some(4.0), Box::new(MUL));


        



    }



}
