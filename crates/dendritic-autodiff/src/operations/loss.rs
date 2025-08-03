use ndarray::{Axis, Array2, stack};
use log::debug; 

use crate::operations::base::*;
use crate::node::{Node}; 
use crate::graph::ComputationGraph; 


/// Shared trait for constructing scalar binary operations.
pub trait LossFunction<T> {

    /// Mean squared error
    fn mse(&mut self, val: T) -> &mut ComputationGraph<T>;

    /// Binary cross entropy
    fn bce(&mut self, val: T) -> &mut ComputationGraph<T>;

    /// Categorical cross entropy
    fn cce(&mut self, val: T) -> &mut ComputationGraph<T>;

    /// Default function for no loss function provided
    fn default(&mut self) -> &mut ComputationGraph<T>;

}

macro_rules! loss_funcs {

    ($t:ty) => {

        impl LossFunction<$t> for ComputationGraph<$t> {

            fn mse(&mut self, val: $t) -> &mut ComputationGraph<$t> {
                self.unary(val, Box::new(MSE))
            }

            fn bce(&mut self, val: $t) -> &mut ComputationGraph<$t> {
                self.unary(val, Box::new(BinaryCrossEntropy))
            }

            fn cce(&mut self, val: $t) -> &mut ComputationGraph<$t> {
                self.unary(val, Box::new(CategoricalCrossEntropy))
            }

            fn default(&mut self) -> &mut ComputationGraph<$t> {
                self.function(Box::new(DefaultLossFunction))
            }

        }

    }
}

loss_funcs!(f64); 
loss_funcs!(Array2<f64>); 


#[derive(Clone, Debug)]
pub struct DefaultLossFunction;

impl Operation<Array2<f64>> for DefaultLossFunction {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug!(
            "Performing forward default loss on node: {:?}",
            curr_idx
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

        debug!(
            "Performing backward default loss on node index: {:?}",
            curr_idx
        );

        let grad = nodes[curr_idx].output();
        nodes[curr_idx].set_grad_output(grad.clone());

        for idx in nodes[curr_idx].inputs() {
            nodes[idx].set_grad_output(grad.clone()); 
        }

        debug!(
            "Updated gradients for node input indexes: {:?}",
            nodes[curr_idx].inputs()
        ); 

    }
}


impl Operation<f64> for DefaultLossFunction {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug!(
            "Performing forward default function on node index: {:?}",
            curr_idx
        ); 

        nodes[curr_idx].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {


        debug!(
            "Performing backward default loss on node index: {:?}",
            curr_idx
        );

        let grad = nodes[curr_idx].output();
        nodes[curr_idx].set_grad_output(grad);

        debug!(
            "Updated gradients for node input indexes: {:?}",
            nodes[curr_idx].inputs()
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

        debug!(
            "Performing forward MSE on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output();
        let y_true = nodes[inputs[1]].output();

        debug!(
            "MSE shape comparison: {:?} {:?}",
            y_true.shape(), y_pred.shape()
        );

        let diff = y_true.clone() - y_pred.clone();
        let squared = diff.mapv(|x| x * x); 
        let sum = squared.sum(); 
        let val = sum * (1.0/y_true.len() as f64);
        Array2::from_elem((1, 1), val) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug!(
            "Performing backward MSE on node index: {:?}",
            curr_idx
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_idx].set_grad_output(grad.clone());
        nodes[inputs[0]].set_grad_output(grad.clone());
        nodes[inputs[1]].set_grad_output(grad);

        debug!(
            "Updated gradients for node input indexes: {:?}",
            inputs
        ); 

    }
}


impl Operation<f64> for MSE {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug!(
            "Performing forward MSE on node index: {:?}",
            curr_idx
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


        debug!(
            "Performing backward multiply on node index: {:?}",
            curr_idx
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_idx].set_grad_output(grad);
        nodes[inputs[0]].set_grad_output(grad);
        nodes[inputs[1]].set_grad_output(grad);

        debug!(
            "Updated gradients for node input indexes: {:?}",
            inputs
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

        debug!(
            "Performing forward BCE on node index: {:?}",
            curr_idx
        ); 

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        if y_pred.shape() != y_true.shape() {
            panic!(
                "Value shapes for binary cross entropy not equal {:?} != {:?}",
                y_pred.shape(), y_true.shape()
            );
        }

        // shape validation
        let mut idx = 0; 
        let mut result = 0.0;
        for y in y_true.iter() {
            let y_val = y_pred[(idx, 0)];
            let diff = -(y * y_val.ln() + (1.0 - y) * (1.0-y_val).ln()); 
            result += diff; 
            idx += 1;
        } 

        result /= y_true.len() as f64; 
        Array2::from_elem((1, 1), result) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug!(
            "Performing backward BCE on node index: {:?}",
            curr_idx
        );

        let inputs = nodes[curr_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();

        let epsilon = 1e-7;
        let mut grad = Array2::<f64>::zeros(y_pred.raw_dim());

        for ((g, &y_t), &y_p_raw) in grad.iter_mut()
            .zip(y_true.iter())
            .zip(y_pred.iter()) 
        {
            let y_p = y_p_raw.clamp(epsilon, 1.0 - epsilon);
            *g = -(y_t / y_p) + (1.0 - y_t) / (1.0 - y_p);
        }

        nodes[curr_idx].set_grad_output(grad); 

        debug!(
            "Updated gradients for node input indexes: {:?}",
            inputs
        ); 

    }
}


impl Operation<f64> for BinaryCrossEntropy {

    fn forward(
        &self, 
        _nodes: &Vec<Node<f64>>, 
        _curr_idx: usize) -> f64 {

        debug!("BCE for scalar values not implemented yet..");
        unimplemented!();

    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<f64>>, 
        _curr_idx: usize) {

        debug!("BCE for scalar values not implemented yet..");
        unimplemented!();

    }
}


#[derive(Clone, Debug)]
pub struct CategoricalCrossEntropy;


impl Operation<Array2<f64>> for CategoricalCrossEntropy {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        let inputs = nodes[curr_idx].inputs();
        let logits = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();

        debug!("INPUTS: {:?}", inputs); 

        let softmax_samples: Vec<_> = logits
            .axis_iter(Axis(0))
            .map(|row| {
                let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp = row.mapv(|x| (x - max).exp());
                let sum = exp.sum();
                exp.mapv(|x| x / sum)
            })
            .collect();

        let views: Vec<_> = softmax_samples.iter().map(|r| r.view()).collect();
        let softmax = stack(Axis(0), &views).unwrap();

        // calculate loss
        let mut loss = 0.0;
        for ((i, j), y) in y_true.indexed_iter() {
            let diff = -y * softmax[[i, j]].ln();
            loss += diff; 
        }

        debug!("Performing forward CCE on node: {:?}", curr_idx);
        let batch_size = y_true.nrows() as f64; 
        let total_loss = loss / batch_size;
        Array2::from_elem((1, 1), total_loss)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        let inputs = nodes[curr_idx].inputs();
        let logits = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let n = y_true.clone().len();

        if logits.shape() != y_true.shape() {
            panic!("Value shapes for categorical cross entropy not equal");
        }  

        let softmax_samples: Vec<_> = logits
            .axis_iter(Axis(0))
            .map(|row| {
                let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp = row.mapv(|x| (x - max).exp());
                let sum = exp.sum();
                exp.mapv(|x| x / sum)
            })
            .collect();

        let views: Vec<_> = softmax_samples.iter().map(|r| r.view()).collect();
        let softmax = stack(Axis(0), &views).unwrap();

        // subtract y_pred from y_true
        let grad = softmax.clone() - y_true;

        nodes[curr_idx].set_grad_output(grad.clone());
        nodes[inputs[1]].set_grad_output(softmax); // opened issue for how to solve this later on
    }
}


impl Operation<f64> for CategoricalCrossEntropy {

    fn forward(
        &self, 
        _nodes: &Vec<Node<f64>>, 
        _curr_idx: usize) -> f64 {

        debug!("CCE for scalar values not implemented yet..");
        unimplemented!();

    }

    fn backward(
        &self, 
        _nodes: &mut Vec<Node<f64>>, 
        _curr_idx: usize) {

        debug!("CCE for scalar values not implemented yet..");
        unimplemented!();

    }
}
