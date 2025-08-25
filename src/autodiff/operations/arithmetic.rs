//! Standard arithmetic operations

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
                let final_grad = grad.clone().into_shape_with_order((1, grad.dim())).unwrap();
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


#[cfg(test)]
mod arithmetic_ops_test {

    use crate::autodiff::graph::*;
    use crate::autodiff::operations::arithmetic::*; 
    use crate::autodiff::operations::loss::*; 
    use ndarray::{arr2};

    #[test]
    fn test_add() {

        let mut scalar_graph = ComputationGraph::new();
        scalar_graph.add(vec![2.0, 3.0]);
        scalar_graph.add(vec![4.0]);
    
        assert_eq!(scalar_graph.nodes().len(), 5); 

        assert_eq!(scalar_graph.node(0).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(0).output(), 2.0);
        assert_eq!(scalar_graph.node(0).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(1).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(1).output(), 3.0); 
        assert_eq!(scalar_graph.node(1).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(2).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(2).output(), 0.0);
        assert_eq!(scalar_graph.node(2).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(3).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(3).output(), 4.0);
        assert_eq!(scalar_graph.node(3).upstream(), vec![4]);
 
        assert_eq!(scalar_graph.node(4).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(4).output(), 0.0);
        assert_eq!(scalar_graph.node(4).upstream().len(), 0);

        scalar_graph.forward();

        assert_eq!(scalar_graph.node(2).output(), 5.0); 
        assert_eq!(scalar_graph.node(4).output(), 9.0);

        scalar_graph.backward();

        assert_eq!(scalar_graph.node(3).grad(), 1.0); 
        assert_eq!(scalar_graph.node(2).grad(), 1.0); 
        assert_eq!(scalar_graph.node(1).grad(), 1.0); 
        assert_eq!(scalar_graph.node(0).grad(), 1.0); 

        let a = arr2(&[[1.0], [2.0], [3.0]]); 
        let b = arr2(&[[1.0], [2.0], [3.0]]); 
        let c = arr2(&[[1.0], [1.0], [1.0]]); 

        let mut nd_graph = ComputationGraph::new();
        nd_graph.add(vec![a.clone(), b.clone()]);
        nd_graph.add(vec![c.clone()]);
        nd_graph.default(); 

        assert_eq!(nd_graph.nodes().len(), 6); 
        assert_eq!(nd_graph.path().len(), 0); 

        nd_graph.forward();

        assert_eq!(nd_graph.node(2).output().shape(), vec![3, 1]); 
        assert_eq!(
            nd_graph.node(2).output(),
            arr2(&[[2.0],[4.0],[6.0]])
        );

        assert_eq!(nd_graph.node(4).output().shape(), vec![3, 1]); 
        assert_eq!(
            nd_graph.node(4).output(),
            arr2(&[[3.0],[5.0],[7.0]])
        );

        nd_graph.backward();

        let vars = nd_graph.variables();

        for (_idx, var) in vars.iter().enumerate() {
            println!("{:?}", nd_graph.node(*var).grad()); 
            /*
            assert_eq!(
                nd_graph.node(*var).grad(),
                expected_grads[idx]
            ); */
        } 
    }

    #[test]
    fn test_subtract() {

        let mut scalar_graph = ComputationGraph::new();
        scalar_graph.sub(vec![10.0, 5.0]);
        scalar_graph.sub(vec![2.0]);
    
        assert_eq!(scalar_graph.nodes().len(), 5); 

        assert_eq!(scalar_graph.node(0).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(0).output(), 10.0);
        assert_eq!(scalar_graph.node(0).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(1).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(1).output(), 5.0); 
        assert_eq!(scalar_graph.node(1).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(2).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(2).output(), 0.0);
        assert_eq!(scalar_graph.node(2).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(3).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(3).output(), 2.0);
        assert_eq!(scalar_graph.node(3).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(4).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(4).output(), 0.0);
        assert_eq!(scalar_graph.node(4).upstream().len(), 0);

        scalar_graph.forward();

        assert_eq!(scalar_graph.node(2).output(), 5.0); 
        assert_eq!(scalar_graph.node(4).output(), 3.0);

        scalar_graph.backward();

        let vars = scalar_graph.variables();
        for (_idx, var) in vars.iter().enumerate() {
            assert_eq!(scalar_graph.node(*var).grad(), 1.0);
        }

    } 

    #[test]
    fn test_multiply() {

        let mut scalar_graph = ComputationGraph::new();
        scalar_graph.mul(vec![10.0, 5.0]);
        scalar_graph.mul(vec![2.0]);
    
        assert_eq!(scalar_graph.nodes().len(), 5); 

        assert_eq!(scalar_graph.node(0).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(0).output(), 10.0);
        assert_eq!(scalar_graph.node(0).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(1).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(1).output(), 5.0); 
        assert_eq!(scalar_graph.node(1).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(2).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(2).output(), 0.0);
        assert_eq!(scalar_graph.node(2).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(3).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(3).output(), 2.0);
        assert_eq!(scalar_graph.node(3).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(4).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(4).output(), 0.0);
        assert_eq!(scalar_graph.node(4).upstream().len(), 0);

        scalar_graph.forward();

        assert_eq!(scalar_graph.node(2).output(), 50.0); 
        assert_eq!(scalar_graph.node(4).output(), 100.0);

        scalar_graph.backward();

        let a = arr2(&[
            [1.0, 1.0, 1.0], 
            [2.0, 2.0, 2.0], 
            [3.0, 3.0, 3.0]
        ]); 
        let b = arr2(&[[1.0], [2.0], [3.0]]); 
        let c = arr2(&[[1.0]]); 

        let mut nd_graph = ComputationGraph::new();
        nd_graph.mul(vec![a.clone(), b.clone()]);
        nd_graph.mul(vec![c.clone()]);
        nd_graph.default(); 

        assert_eq!(nd_graph.nodes().len(), 6); 
        assert_eq!(nd_graph.path().len(), 0); 

        nd_graph.forward();

        assert_eq!(nd_graph.node(2).output().shape(), vec![3, 1]); 
        
        assert_eq!(
            nd_graph.node(2).output(),
            arr2(&[[6.0],[12.0],[18.0]])
        );

        assert_eq!(nd_graph.node(4).output().shape(), vec![3, 1]); 
        assert_eq!(
            nd_graph.node(4).output(),
            arr2(&[[6.0],[12.0],[18.0]])
        );

        nd_graph.backward(); 

        let grad_1 = arr2(&[
            [6.0, 12.0, 18.0],
            [12.0, 24.0, 36.0],
            [18.0, 36.0, 54.0]
        ]);
        let grad_2 = arr2(&[[84.0], [84.0], [84.0]]);
        let grad_3 = arr2(&[[504.0]]);
        let grads = vec![grad_1, grad_2, grad_3]; 

        let vars = nd_graph.variables();
        for (idx, var) in vars.iter().enumerate() {
            assert_eq!(
                nd_graph.node(*var).grad(),
                grads[idx]
            );
        }
    }

}
