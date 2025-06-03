use ndarray::Array2;
use std::fmt::Debug; 

use crate::graph::*; 
use crate::ops::*;
use crate::node::{Node}; 


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


/// Shared trait for constructing scalar binary operations.
pub trait LossFunction<T> {

    /// Mean squared error
    fn mse(&mut self, val: T) -> &mut ComputationGraph<T>;

    /// Binary cross entropy
    fn bce(&mut self, val: T) -> &mut ComputationGraph<T>;

    /// Default function for no loss function provided
    fn default(&mut self) -> &mut ComputationGraph<T>;

}


impl LossFunction<Array2<f64>> for ComputationGraph<Array2<f64>> {

    fn mse(&mut self, val: Array2<f64>) -> &mut ComputationGraph<Array2<f64>> {
        self.unary(val, Box::new(MSE))
    }

    fn bce(&mut self, val: Array2<f64>) -> &mut ComputationGraph<Array2<f64>> {
        self.unary(val, Box::new(BinaryCrossEntropy))
    }

    fn default(&mut self) -> &mut ComputationGraph<Array2<f64>> {

        let curr_node = self.curr_node_idx as usize;
        let prev_node = self.nodes[self.curr_node_idx as usize].clone(); 
        self.add_node(
            Node::unary(curr_node, Box::new(DefaultLossFunction))
        );

        let new_node_idx = self.curr_node_idx as usize;
        self.nodes[new_node_idx].set_output(prev_node.output());
        self.nodes[new_node_idx].set_grad_output(prev_node.output());

        self.add_upstream_node(
            curr_node, 
            vec![self.curr_node_idx as usize]
        );

        self
    }

}


/// Shared trait for constructing scalar binary operations.
pub trait ActivationFunction<T> {

    /// Sigmoid activation function for non linear data
    fn sigmoid(&mut self) -> &mut ComputationGraph<T>;

    /// Tanh activation function 
    fn tanh(&mut self) -> &mut ComputationGraph<T>;

}


impl ActivationFunction<Array2<f64>> for ComputationGraph<Array2<f64>> {

    fn sigmoid(&mut self) -> &mut ComputationGraph<Array2<f64>> {

        let curr_node = self.curr_node_idx as usize;
        let prev_node = self.nodes[self.curr_node_idx as usize].clone(); 
        self.add_node(
            Node::unary(curr_node, Box::new(Sigmoid))
        );

        let new_node_idx = self.curr_node_idx as usize;
        self.nodes[new_node_idx].set_output(prev_node.output());
        self.nodes[new_node_idx].set_grad_output(prev_node.output());

        self.add_upstream_node(
            curr_node, 
            vec![self.curr_node_idx as usize]
        );
        self
    }

    fn tanh(&mut self) -> &mut ComputationGraph<Array2<f64>> {

        let curr_node = self.curr_node_idx as usize;
        let prev_node = self.nodes[self.curr_node_idx as usize].clone(); 
        self.add_node(
            Node::unary(curr_node, Box::new(Tanh))
        );

        let new_node_idx = self.curr_node_idx as usize;
        self.nodes[new_node_idx].set_output(prev_node.output());
        self.nodes[new_node_idx].set_grad_output(prev_node.output());

        self.add_upstream_node(
            curr_node, 
            vec![self.curr_node_idx as usize]
        );
        self
    } 

}
