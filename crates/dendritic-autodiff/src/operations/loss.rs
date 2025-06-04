use std::fmt; 
use std::fmt::{Debug, Display};
use crate::operations::base::*;
use crate::tensor::Tensor;
use crate::node::{Node}; 
use crate::graph::ComputationGraph; 
use chrono::Local; 
use ndarray::{arr2, Array2};


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


#[derive(Clone, Debug)]
pub struct DefaultLossFunction;

impl Operation<Array2<f64>> for DefaultLossFunction {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward default loss on node: {:?}",
                curr_idx
            ) 
        ); 

        let inputs = nodes[curr_idx].inputs();
        match inputs.len() {
            1 => {
                nodes[inputs[0]].output()
            },
            0 => {
                panic!("Previous node needs to exist to create default loss function");
            },
            _ => {
                panic!("Error validating inputs for default loss function forward pass"); 
            }
        }
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        debug_log(
            &format!(
                "Performing backward default loss on node index: {:?}",
                curr_idx
            ) 
        );

        let grad = nodes[curr_idx].output();
        nodes[curr_idx].set_grad_output(grad.clone());

        for idx in nodes[curr_idx].inputs() {
            nodes[idx].set_grad_output(grad.clone()); 
        }

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

    }
}


impl Operation<f64> for DefaultLossFunction {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug_log(
            &format!(
                "Performing forward MSE on node index: {:?}",
                curr_idx
            ) 
        ); 

        nodes[curr_idx].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "Performing backward default loss on node index: {:?}",
                curr_idx
            ) 
        );

        let grad = nodes[curr_idx].output();
        nodes[curr_idx].set_grad_output(grad);

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct MSE;

impl Operation<Array2<f64>> for MSE {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward MSE on node index: {:?}",
                curr_idx
            ) 
        ); 

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output();
        let y_true = nodes[inputs[1]].output();

        let diff = y_true.clone() - y_pred; 
        let squared = diff.mapv(|x| x * x); 
        let sum = squared.sum(); 
        let val = sum * (1.0/y_true.len() as f64);
        Array2::from_elem((1, 1), val) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "Performing backward MSE on node index: {:?}",
                curr_idx
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_idx].set_grad_output(grad.clone());
        nodes[inputs[0]].set_grad_output(grad.clone());
        nodes[inputs[1]].set_grad_output(grad);

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                inputs
            ) 
        ); 

    }
}


impl Operation<f64> for MSE {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug_log(
            &format!(
                "Performing forward MSE on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "Forward MSE upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output();
        let y_true = nodes[inputs[1]].output();

        let diff = y_true.clone() - y_pred;
        diff.powf(2.0) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_idx].set_grad_output(grad);
        nodes[inputs[0]].set_grad_output(grad);
        nodes[inputs[1]].set_grad_output(grad);

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                inputs
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct BinaryCrossEntropy;

impl Operation<Array2<f64>> for BinaryCrossEntropy {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward MSE on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "Forward MSE upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();

        // shape validation
        let mut idx = 0; 
        let mut result = 0.0;
        for y in y_true.iter() {
            let y_val = y_pred[(idx, 0)];
            let diff = -(y * y_val.ln() + (1.0 - y) * (1.0-y_val).ln()); 
            result += diff; 
            idx += 1;
        } 

        Array2::from_elem((1, 1), result) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "Performing backward BCE on node index: {:?}",
                curr_idx
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_idx].set_grad_output(grad); 

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                inputs
            ) 
        ); 

    }
}
