use std::fmt::Debug;

use ndarray::Array2;
use log::debug; 

use crate::autodiff::operations::base::*; 
use crate::autodiff::node::{Node}; 
use crate::autodiff::graph::ComputationGraph; 


/// Shared trait for constructing scalar binary operations.
pub trait ActivationFunction<T> {

    /// Sigmoid activation function for non linear data
    fn sigmoid(&mut self) -> &mut ComputationGraph<T>;

    /// Tanh activation function 
    fn tanh(&mut self) -> &mut ComputationGraph<T>;

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


#[cfg(test)]
mod activation_ops_test {

    use crate::autodiff::graph::*;
    use crate::autodiff::operations::activation::*; 
    use crate::autodiff::operations::arithmetic::*; 
    use crate::autodiff::operations::loss::*; 
    use ndarray::prelude::*; 
    use ndarray::{arr2};


    #[test]
    fn test_sigmoid() {

        let w = Array2::<f64>::zeros((3, 1));
        let b = Array2::<f64>::zeros((1, 1));

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]); 
        
        let mut graph = ComputationGraph::new();
        graph.mul(vec![x, w]); 
        graph.add(vec![b]);
        graph.sigmoid(); 
        graph.mse(y.clone());

        graph.forward(); 

        let add_output = graph.node(4); 
        let sig_output = graph.node(5); 

        assert_eq!(
            add_output.output(), 
            arr2(&[[0.0],[0.0],[0.0],[0.0], [0.0]])
        );

        assert_eq!(
            sig_output.output(), 
            arr2(&[[0.5],[0.5],[0.5],[0.5], [0.5]])
        );

        graph.backward();

        assert_eq!(
            graph.node(4).grad(),
            arr2(&[[-2.375],[-2.875],[-3.375],[-3.875],[-4.375]])
        );

    }


    #[test]
    fn test_tanh() {

        let b = Array2::<f64>::zeros((1, 1));
        let w = arr2(&[[1.0],[2.0],[3.0]]);

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]); 
        
        let mut graph = ComputationGraph::new();
        graph.mul(vec![x, w]); 
        graph.add(vec![b]);
        graph.tanh(); 
        graph.mse(y.clone());

        graph.forward();

        let add_output = graph.node(4); 
        let tan_output = graph.node(5);

        assert_eq!(
            add_output.output(),
            arr2(&[[14.0],[20.0],[26.0],[32.0],[38.0]])
        );

        assert_eq!(
            tan_output.output(),
            arr2(&[[0.9999999999986171],[1.0],[1.0],[1.0],[1.0]])
        );

        graph.backward(); 

        assert_eq!(
            graph.node(4).grad(),
            arr2(&[
                [-9.000000000001382],
                [-11.0],
                [-13.0],
                [-15.0],
                [-17.0]
            ])
        );

    }

}




