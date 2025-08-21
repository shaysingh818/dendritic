use std::fmt; 
use std::fmt::Debug;

use serde::{Serialize, Deserialize}; 
use ndarray::{Array2, Axis};
use log::debug; 

use crate::autodiff::operations::base::*; 
use crate::autodiff::node::{Node}; 
use crate::autodiff::graph::ComputationGraph; 

/// Shared trait for constructing scalar binary operations.
pub trait Arithmetic<T> {

    fn add(&mut self, inputs: Vec<T>) -> &mut ComputationGraph<T>; 

    fn sub(&mut self, inputs: Vec<T>) -> &mut ComputationGraph<T>; 

    fn mul(&mut self, inputs: Vec<T>) -> &mut ComputationGraph<T>; 

}


macro_rules! arithmetic_ops {

    ($t:ty) => {

        impl Arithmetic<$t> for ComputationGraph<$t> {

            fn add(&mut self, inputs: Vec<$t>) -> &mut ComputationGraph<$t> {
                match inputs.len() {
                    2 => {
                        self.binary(Some(inputs[0].clone()), Some(inputs[1].clone()), Box::new(Add))
                    },

                    1 => {
                        self.unary(inputs[0].clone(), Box::new(Add))
                    },
                    _ => {
                        panic!("add: One or 2 inputs required");
                    }
                }
            }

            fn sub(&mut self, inputs: Vec<$t>) -> &mut ComputationGraph<$t> {

                match inputs.len() {
                    2 => {
                        self.binary(Some(inputs[0].clone()), Some(inputs[1].clone()), Box::new(Sub))
                    },

                    1 => {
                        self.unary(inputs[0].clone(), Box::new(Sub))
                    },
                    _ => {
                        panic!("sub: One or 2 inputs required");
                    }
                }
            }

            fn mul(&mut self, inputs: Vec<$t>) -> &mut ComputationGraph<$t> {

                match inputs.len() {
                    2 => {
                        self.binary(Some(inputs[0].clone()), Some(inputs[1].clone()), Box::new(Mul))
                    },

                    1 => {
                        self.unary(inputs[0].clone(), Box::new(Mul))
                    },
                    _ => {
                        panic!("mul: One or 2 inputs required");
                    }
                }
            }

        }
    }
}

arithmetic_ops!(f64);
arithmetic_ops!(Array2<f64>);


#[derive(Clone, Debug)]
pub struct Add; 

impl Operation<f64> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug!(
            "(ADD SCALAR) Performing forward pass on node index: {:?}",
            curr_idx
        );

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() + nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {

        debug!(
            "(ADD SCALAR) Performing backward on node index: {:?}",
            nodes[curr_idx].inputs()
        ); 

        let node_inputs = nodes[curr_idx].inputs();
        for (idx, _input) in node_inputs.iter().enumerate() {
            nodes[node_inputs[idx]].set_grad_output(1.0);
        }

        debug!(
            "Updated gradients for node input indexes: {:?}",
            node_inputs
        ); 

    }
}


impl Operation<Array2<f64>> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            "(ADD) Performing forward pass on node index: {:?}",
            curr_idx
        );

        let inputs = nodes[curr_idx].inputs();
        let lhs = nodes[inputs[0]].output();
        let rhs = nodes[inputs[1]].output();

        debug!(
            "[ADD]: [Node {:?}]: {:?} + [Node {:?}]: {:?}", 
            inputs[0], lhs.dim(), inputs[1], rhs.dim()
        ); 
        lhs + rhs
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {



        let inputs = nodes[curr_idx].inputs();
        let upstream = nodes[curr_idx].upstream();

        if upstream.len() > 1 {
            panic!("Backward addition can only handle one upstream"); 
        }

        let upstream_grad = nodes[upstream[0]].grad();
        nodes[curr_idx].set_grad_output(upstream_grad.clone());

        debug!(
            "[ADD] Upstream: {:?} Inputs: {:?}", 
            upstream[0], inputs
        ); 

        for input_idx in &inputs {
            let input_shape = nodes[*input_idx].output().dim();
            if input_shape.0 == 1 {
                let grad = upstream_grad.sum_axis(Axis(0));
                let final_grad = grad.clone().into_shape((1, grad.dim())).unwrap();
                nodes[*input_idx].set_grad_output(final_grad);
            } else {
                nodes[*input_idx].set_grad_output(upstream_grad.clone());
            }
        }

        debug!(
            "(ADD) Updated gradients for node input indexes: {:?}",
            inputs
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct Sub; 

impl Operation<f64> for Sub {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug!(
            "Performing forward pass subtract on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() - nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {

        debug!(
            "Performing backward subtract on node index: {:?}",
            curr_idx
        ); 

        let node_inputs = nodes[curr_idx].inputs();
        for (idx, _input) in node_inputs.iter().enumerate() {
            nodes[node_inputs[idx]].set_grad_output(1.0);
        }

        debug!(
            "Updated gradients for node input indexes: {:?}",
            node_inputs
        ); 

    }
}


impl Operation<Array2<f64>> for Sub {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            "Forward subtraction on node: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();
        debug!("[SUB]: {:?} - {:?}", lhs.dim(), rhs.dim()); 
        lhs - rhs
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        debug!(
            "Performing backward multiply on node index: {:?}",
            curr_idx
        ); 

        let node_inputs = nodes[curr_idx].inputs();
        let node_upstream = nodes[curr_idx].upstream(); 

        let lhs = nodes[node_inputs[0]].output(); 
        let rhs = nodes[node_inputs[1]].output();

        if node_upstream.len() > 1 {
            panic!("Backward multiply can only handle one upstream"); 
        }

        let upstream = nodes[node_upstream[0]].output();  
        let rhs_grad = upstream.dot(&rhs.t());
        let lhs_grad = lhs.t().dot(&upstream);
        debug!("[SUB]: Upstream node {:?}", node_upstream[0]); 

        nodes[node_inputs[0]].set_grad_output(rhs_grad); 
        nodes[node_inputs[1]].set_grad_output(lhs_grad);

        debug!(
            "Updated gradients for node input indexes: {:?}",
            node_inputs
        ); 

    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mul;


impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trait: {}", self)
    }
} 

impl Operation<f64> for Mul {

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


impl Operation<Array2<f64>> for Mul {

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
