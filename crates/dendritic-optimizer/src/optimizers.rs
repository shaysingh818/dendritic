use std::io::Write; 

use chrono::Local; 
use ndarray::{arr2, Array2, Axis};

use dendritic_autodiff::node::*; 
use dendritic_autodiff::graph::*;
use dendritic_autodiff::operations::activation::*; 
use dendritic_autodiff::operations::loss::*; 
use crate::regression::logistic::*; 
use crate::regression::sgd::*;
use crate::train::*;
use crate::model::*;


pub trait Optimizer {

    fn step(&mut self, graph: &mut ComputationGraph<Array2<f64>>); 

    fn reset(&mut self);

    fn parameters(&self) -> Vec<usize>;

}

pub struct Adam {

    /// Learning rate associated with model
    alpha: f64,

    /// Epsilon to represent small value
    epsilon: f64, 

    /// First gradient decay
    y_v: f64,

    /// Second gradient decay
    y_s: f64,

    /// Step counter (iteration count)
    k: usize,

    /// First momentum estimation
    v_delta: Vec<Array2<f64>>,

    /// Second momentum estimation
    s_delta: Vec<Array2<f64>>
    
}


impl Adam {

    pub fn default(
        learning_rate: f64,
        parameter_idxs: Vec<usize>,
        nodes: Vec<Node<Array2<f64>>>) -> Self {

        let mut obj = Adam {
            alpha: learning_rate,
            epsilon: 1e-6,
            y_v: 0.9,
            y_s: 0.999,
            k: 0, 
            v_delta: Vec::new(),
            s_delta: Vec::new()
        };

        obj.parameter_momentum_init(parameter_idxs, nodes);
        obj

    }

    pub fn grad_decay_1(&self) -> f64 {
        self.y_v
    }
        
    pub fn grad_decay_2(&self) -> f64 {
        self.y_s
    }

    pub fn step_count(&self) -> usize {
        self.k
    }

    pub fn first_momentum(&self) -> Vec<Array2<f64>> {
        self.v_delta.clone()
    }

    pub fn second_momentum(&self) -> Vec<Array2<f64>> {
        self.v_delta.clone()
    }

    pub fn parameter_momentum_init(
        &mut self,
        parameter_idxs: Vec<usize>,
        nodes: Vec<Node<Array2<f64>>>) {

        for param in parameter_idxs {
            let parameter_node = &nodes[param];
            let parameter_shape = parameter_node.output().dim();
            self.v_delta.push(Array2::zeros(parameter_shape));
            self.s_delta.push(Array2::zeros(parameter_shape));
        }
    }

}


impl Optimizer for Adam {

    fn step(&mut self, graph: &mut ComputationGraph<Array2<f64>>) {
        let params = graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {

            let parameter = graph.node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);

            self.v_delta[idx] = self.y_v * self.v_delta[idx].clone() + (1.0 - self.y_v) * grad.clone(); 
            self.s_delta[idx] = self.y_s * self.s_delta[idx].clone() + (1.0 - self.y_s) * grad;
            self.k += 1;

            let v_hat = self.v_delta[idx].clone() / (1.0 - self.y_v.powf(self.k as f64)); 
            let s_hat = self.s_delta[idx].clone() / (1.0 - self.y_s.powf(self.k as f64)); 
            let param_delta = self.alpha * v_hat / (s_hat.mapv(f64::sqrt) + self.epsilon); 

            graph.mut_node_output(param, parameter.output() - param_delta); 

            
            
        }
    }

    fn reset(&mut self) {
        println!("Testing reset.."); 
    }

    fn parameters(&self) -> Vec<usize> {
        vec![0, 1, 2]
    }   

}


