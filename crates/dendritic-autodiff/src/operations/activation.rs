use std::fmt; 
use std::fmt::{Debug, Display};


use ndarray::{arr2, Array2};
use log::{debug, warn, info}; 

use crate::operations::base::*; 
use crate::tensor::Tensor;
use crate::node::{Node}; 
use crate::graph::ComputationGraph; 


/// Shared trait for constructing scalar binary operations.
pub trait ActivationFunction<T> {

    /// Sigmoid activation function for non linear data
    fn sigmoid(&mut self) -> &mut ComputationGraph<T>;

    /// Tanh activation function 
    fn tanh(&mut self) -> &mut ComputationGraph<T>;

}


impl ActivationFunction<Array2<f64>> for ComputationGraph<Array2<f64>> {

    fn sigmoid(&mut self) -> &mut ComputationGraph<Array2<f64>> {
        self.function(Box::new(Sigmoid))
    }

    fn tanh(&mut self) -> &mut ComputationGraph<Array2<f64>> {
        self.function(Box::new(Tanh))
    } 

}


#[derive(Clone, Debug)]
pub struct Sigmoid;

impl Operation<Array2<f64>> for Sigmoid {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            &format!(
                "Sigmoid activation on node index: {:?}",
                curr_idx
            ) 
        ); 

        let inputs = nodes[curr_idx].inputs();
        if inputs.len() != 1 {
            panic!("Sigmoid node can only handle 1 input"); 
        }

        let input = nodes[inputs[0]].output();
        input.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug!(
            &format!(
                "Performing backward sigmoid on node index: {:?}",
                curr_idx
            ) 
        );


        let inputs = nodes[curr_idx].inputs();
        if inputs.len() != 1 {
            panic!("Sigmoid node can only handle 1 input"); 
        }

        let upstream = nodes[curr_idx].upstream();

        fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + f64::exp(-x)) }

        match upstream.len() {
            1 => {

                let upstream = nodes[upstream[0]].grad();
                let input = nodes[inputs[0]].output();

                let sig: Array2<f64> = input.mapv(
                    |x| sigmoid(x) * (1.0 - sigmoid(x))
                );

                let grad = upstream * sig;
                nodes[inputs[0]].set_grad_output(grad.clone());
            },
            _ => {
                panic!("Sigmoid must only have 1 input"); 
            }
        }

        debug!(
            &format!(
                "Updated gradients for sigmoid operation: {:?}",
                inputs
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct Tanh;

impl Operation<Array2<f64>> for Tanh {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            &format!(
                "Performing TANH on node index: {:?}",
                curr_idx
            ) 
        ); 


        let inputs = nodes[curr_idx].inputs();
        if inputs.len() != 1 {
            panic!("TANH node can only handle 1 input"); 
        }

        let input = nodes[inputs[0]].output();
        input.mapv(|v| v.tanh())
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug!(
            &format!(
                "Performing backward TANH on node index: {:?}",
                curr_idx
            ) 
        );


        let inputs = nodes[curr_idx].inputs();
        if inputs.len() != 1 {
            panic!("TANH node can only handle 1 input"); 
        }

        let upstream = nodes[curr_idx].upstream();
        match upstream.len() {
            1 => {

                let upstream = nodes[upstream[0]].grad();
                let input = nodes[inputs[0]].output(); 
                
                let tan: Array2<f64> = input.mapv(
                    |x| 1.0 - x.tanh().powf(2.0)
                );

                //println!("Shape debugging"); 
                //println!("Shape: {:?}, upstream idx: {:?}", upstream.shape(), upstream); 
                //println!("{:?}", tan.shape()); 

                let grad = upstream * tan;
                nodes[inputs[0]].set_grad_output(grad.clone());
            },
            _ => {
                panic!("TANH must only have 1 upstream value"); 
            }
        }


        debug!(
            &format!(
                "Updated gradients for TANH operation: {:?}",
                inputs
            ) 
        ); 

    }
}
