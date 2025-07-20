use std::fmt::Debug;

use ndarray::Array2;
use log::debug; 

use crate::operations::base::*; 
use crate::node::{Node}; 
use crate::graph::ComputationGraph; 


/// Shared trait for constructing scalar binary operations.
pub trait ActivationFunction<T> {

    /// Sigmoid activation function for non linear data
    fn sigmoid(&mut self) -> &mut ComputationGraph<T>;

    /// Tanh activation function 
    fn tanh(&mut self) -> &mut ComputationGraph<T>;

    /// Softmax activation function, typically for multi class classification
    fn softmax(&mut self) -> &mut ComputationGraph<T>;

}

macro_rules! activation_funcs {

    ($t:ty) => {

        impl ActivationFunction<$t> for ComputationGraph<$t> {

            fn sigmoid(&mut self) -> &mut ComputationGraph<$t> {
                self.function(Box::new(Sigmoid))
            }

            fn tanh(&mut self) -> &mut ComputationGraph<$t> {
                self.function(Box::new(Tanh))
            }

            fn softmax(&mut self) -> &mut ComputationGraph<$t> {
                self.function(Box::new(Softmax))
            } 

        }
    }

}

activation_funcs!(f64); 
activation_funcs!(Array2<f64>); 


#[derive(Clone, Debug)]
pub struct Sigmoid;

impl Operation<Array2<f64>> for Sigmoid {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            "Sigmoid activation on node index: {:?}",
            curr_idx
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
            "Performing backward sigmoid on node index: {:?}",
            curr_idx
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
                let sig_output = nodes[curr_idx].output();
                let sig_deriv = sig_output.mapv(|s| s * (1.0 - s));
                let grad = upstream * &sig_deriv;

                nodes[curr_idx].set_grad_output(grad.clone());
                nodes[inputs[0]].set_grad_output(grad.clone()); 
            },
            _ => {
                panic!("Sigmoid must only have 1 input"); 
            }
        }

        debug!(
            "Updated gradients for sigmoid operation: {:?}",
            inputs
        ); 

    }
}

impl Operation<f64> for Sigmoid {

    fn forward(
        &self, 
        _nodes: &Vec<Node<f64>>, 
        _curr_idx: usize) -> f64 {

        debug!("Sigmoid for scalar values not implemented yet..");
        unimplemented!();

    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<f64>>, 
        _curr_idx: usize) {

        debug!("Sigmoid for scalar values not implemented yet..");
        unimplemented!();

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
            "Performing TANH on node index: {:?}",
            curr_idx
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
            "Performing backward TANH on node index: {:?}",
            curr_idx
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

                let grad = upstream * tan;
                nodes[inputs[0]].set_grad_output(grad.clone());
            },
            _ => {
                panic!("TANH must only have 1 upstream value"); 
            }
        }


        debug!(
            "Updated gradients for TANH operation: {:?}",
            inputs
        ); 

    }
}


impl Operation<f64> for Tanh {

    fn forward(
        &self, 
        _nodes: &Vec<Node<f64>>, 
        _curr_idx: usize) -> f64 {

        debug!("Tanh for scalar values not implemented yet..");
        unimplemented!();

    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<f64>>, 
        _curr_idx: usize) {

        debug!("Tanh for scalar values not implemented yet..");
        unimplemented!();

    }
}


#[derive(Clone, Debug)]
pub struct Softmax;


impl Operation<f64> for Softmax {

    fn forward(
        &self, 
        _nodes: &Vec<Node<f64>>, 
        _curr_idx: usize) -> f64 {

        debug!("Softmax for scalar values not implemented yet..");
        unimplemented!();

    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<f64>>, 
        _curr_idx: usize) {

        debug!("Softmax for scalar values not implemented yet..");
        unimplemented!();

    }
}


impl Operation<Array2<f64>> for Softmax {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        let inputs = nodes[curr_idx].inputs();
        if inputs.len() != 1 {
            panic!("SOFTMAX node can only handle 1 input"); 
        }

        debug!("Performing forward softmax activation on node: {:?}", curr_idx); 
        let input = nodes[inputs[0]].output();
        let exp = input.mapv(|x| x.exp());
        let sum = exp.sum();
        exp.mapv(|x| x / sum)
    
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        let y = nodes[curr_idx].output();
        let n = y.len();
        let mut jacobian = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    jacobian[[i, j]] = y[[i,0]] * (1.0 - y[[i,0]]);
                } else {
                    jacobian[[i, j]] = -y[[i,0]] * y[[i,0]]; 
                }
            }
        }

        let inputs = nodes[curr_idx].inputs();
        if inputs.len() != 1 {
            panic!("SOFTMAX node can only handle 1 input"); 
        }

        let upstream = nodes[curr_idx].upstream();
        match upstream.len() {
            1 => {
                let upstream_val = nodes[upstream[0]].grad();
                let grad_input = jacobian.dot(&upstream_val);
                debug!("SOFTMAX UPSTREAM GRAD: {:?}", upstream_val); 
                debug!("SOFTMAX GRAD: {:?}", grad_input); 
                nodes[curr_idx].set_grad_output(grad_input.clone());
                nodes[inputs[0]].set_grad_output(grad_input);
            },
            _ => {
                panic!("TANH must only have 1 upstream value"); 
            }
        }
    }

}


#[derive(Clone, Debug)]
pub struct SoftmaxCCE;


impl Operation<f64> for SoftmaxCCE {

    fn forward(
        &self, 
        _nodes: &Vec<Node<f64>>, 
        _curr_idx: usize) -> f64 {

        debug!("SoftmaxFused for scalar values not implemented yet..");
        unimplemented!();

    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<f64>>, 
        _curr_idx: usize) {

        debug!("SoftmaxFused for scalar values not implemented yet..");
        unimplemented!();

    }
}


impl Operation<Array2<f64>> for SoftmaxCCE {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        let inputs = nodes[curr_idx].inputs();

        debug!(
            "Performing forward softmax activation on node: {:?}", 
            curr_idx
        );

        let input = nodes[inputs[0]].output();
        let exp = input.mapv(|x| x.exp());
        let sum = exp.sum();
        exp.mapv(|x| x / sum) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        let inputs = nodes[curr_idx].inputs();
        let softmax = nodes[curr_idx].output();   
        let upstream = nodes[curr_idx].upstream();


        match upstream.len() {

            1 => {
                let y_true = nodes[upstream[0]].grad();
                debug!("SOFTMAX BACK: {:?}", softmax); 
                debug!("Y TRUE: {:?}", y_true); 
                let grad = softmax - y_true; 
                nodes[curr_idx].set_grad_output(grad.clone());
                nodes[inputs[0]].set_grad_output(grad);
            }
            _ => {
                panic!("SOFTMAX must only have 1 upstream value"); 
            }
            
        }
    

    }

}
