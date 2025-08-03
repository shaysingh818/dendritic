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

    fn step(&mut self, model: &mut SGD); 

    fn reset(&mut self, model: &mut SGD);

}

pub struct DefaultOptimizer {

    /// Learning rate associated with model
    alpha: f64

}

impl Optimizer for DefaultOptimizer {

    fn step(&mut self, model: &mut SGD) {
        let params = model.graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph.node(param);
            let grad = parameter.grad() * model.learning_rate;
            let delta = parameter.output() - grad;
            model.graph.mut_node_output(param, delta); 
        }
    }

    fn reset(&mut self, model: &mut SGD) {
        println!("Testing reset.."); 
    }

}


pub struct Nesterov {

    /// Learning rate associated with model
    alpha: f64,
    
    /// Momentum factor associated with model
    beta: f64,

    /// Velocity associated with paramters
    v: Vec<Array2<f64>>

}

impl Nesterov {

    pub fn default(model: &SGD) -> Self {

        let mut velocity_vector: Vec<Array2<f64>> = Vec::new();
        for param in model.graph.parameters() {
            let parameter_node = model.graph.node(param);
            let parameter_shape = parameter_node.output().dim();
            velocity_vector.push(Array2::zeros(parameter_shape));
        }

        Nesterov {
            alpha: model.learning_rate,
            beta: 0.9,
            v: velocity_vector
        }
        
    }

}

impl Optimizer for Nesterov {

    fn step(&mut self, model: &mut SGD) {
        let params = model.graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph.node(param);
            let lookahead = parameter.output() - (self.beta * self.v[idx].clone());
            model.graph.mut_node_output(param, lookahead);

            let grad = parameter.grad() * self.alpha;
            self.v[idx] = grad + (self.beta * self.v[idx].clone());
            let new_param = parameter.output() - self.v[idx].clone();
            model.graph.mut_node_output(param, new_param);
        }
    }

    fn reset(&mut self, model: &mut SGD) {
        println!("Testing reset.."); 
    }

}


pub struct Adagrad {

    /// Learning rate associated with model
    alpha: f64,
    
    /// Momentum factor associated with model
    epsilon: f64,

    /// Not known yet
    s: Vec<Array2<f64>>

}

impl Adagrad {

    pub fn default(model: &SGD) -> Self {

        let mut s_vector: Vec<Array2<f64>> = Vec::new();
        for param in model.graph.parameters() {
            let parameter_node = model.graph.node(param);
            let parameter_shape = parameter_node.output().dim();
            s_vector.push(Array2::zeros(parameter_shape));
        }

        Adagrad {
            alpha: model.learning_rate,
            epsilon: 1e-8,
            s: s_vector
        }
    }

}


impl Optimizer for Adagrad {

    fn step(&mut self, model: &mut SGD) {
        let params = model.graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph.node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);
            self.s[idx] += &grad_squared;

            let ada = self.alpha / (self.s[idx].mapv(f64::sqrt) + self.epsilon);
            let param_new = parameter.output() - (ada * grad); 
            model.graph.mut_node_output(param, param_new);

        }
    }

    fn reset(&mut self, model: &mut SGD) {
        println!("Testing reset.."); 
    }

}


pub struct RMSProp {

    /// Learning rate associated with model
    alpha: f64,
    
    /// Momentum factor associated with model
    epsilon: f64,

    /// Not known yet
    decay_rate: f64,

    /// Not known yet
    s: Vec<Array2<f64>>

}


impl RMSProp {

    pub fn default(model: &SGD) -> Self {

        let mut s_vector: Vec<Array2<f64>> = Vec::new();
        for param in model.graph.parameters() {
            let parameter_node = model.graph.node(param);
            let parameter_shape = parameter_node.output().dim();
            s_vector.push(Array2::zeros(parameter_shape));
        }

        RMSProp {
            alpha: model.learning_rate,
            epsilon: 1e-8,
            decay_rate: 0.9,
            s: s_vector
        }
    }

}


impl Optimizer for RMSProp {

    fn step(&mut self, model: &mut SGD) {
        let params = model.graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph.node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);

            self.s[idx] = self.decay_rate * self.s[idx].clone() + (1.0 - self.decay_rate) * grad_squared;
            let rms = self.alpha / (self.s[idx].mapv(f64::sqrt) + self.epsilon);
            let new = parameter.output() - (rms * grad); 
            model.graph.mut_node_output(param, new);
        }
    }

    fn reset(&mut self, model: &mut SGD) {
        println!("Testing reset.."); 
    }

}


pub struct Adadelta {

    /// Learning rate associated with model
    y_s: f64,
   
    /// Learning rate associated with model
    y_x: f64,

    /// Momentum factor associated with model
    epsilon: f64,

    /// Not known yet
    s: Vec<Array2<f64>>

    /// Not known yet
    u: Vec<Array2<f64>>

}


impl Adadelta {

    pub fn default(model: &SGD) -> Self {

        let mut s_vector: Vec<Array2<f64>> = Vec::new();
        let mut u_vector: Vec<Array2<f64>> = Vec::new();
        for param in model.graph.parameters() {
            let parameter_node = model.graph.node(param);
            let parameter_shape = parameter_node.output().dim();
            s_vector.push(Array2::zeros(parameter_shape)); 
            u_vector.push(Array2::zeros(parameter_shape));
        }

        Adadelta {
            y_s: 0.95,
            y_x: 0.95,
            epsilon: 1e-6,
            s: s_vector,
            u: u_vector
        }
    }
}


impl Optimizer for Adadelta {

    fn step(&mut self, model: &mut SGD) {
        let params = model.graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph.node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);
            
            self.s[idx] = self.y_s * self.s[idx].clone() + (1.0 - self.y_s) * grad_squared;
            let delta = ((self.u[idx].mapv(f64::sqrt) + self.epsilon) / (self.s[idx].mapv(f64::sqrt) + epsilon)) * grad;
            self.u[idx] = self.y_x * self.u[idx].clone() + (1.0 - self.y_x) * delta.mapv(|x| x * x); 
            let new = parameter.output() + (delta * -1.0);
            model.graph.mut_node_output(param, new); 
        }
    }

    fn reset(&mut self, model: &mut SGD) {
        println!("Testing reset.."); 
    }

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

    pub fn default(model: &SGD) -> Self {

        let mut obj = Adam {
            alpha: model.learning_rate,
            epsilon: 1e-6,
            y_v: 0.9,
            y_s: 0.999,
            k: 0, 
            v_delta: Vec::new(),
            s_delta: Vec::new()
        };

        obj.parameter_momentum_init(
            model.graph.parameters(), 
            model.graph.nodes()
        );
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

    fn step(&mut self, model: &mut SGD) {
        let params = model.graph.parameters();
        for (idx, param) in params.into_iter().enumerate() {

            let parameter = model.graph.node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);

            self.v_delta[idx] = self.y_v * self.v_delta[idx].clone() + (1.0 - self.y_v) * grad.clone(); 
            self.s_delta[idx] = self.y_s * self.s_delta[idx].clone() + (1.0 - self.y_s) * grad_squared;
            self.k += 1;

            let v_hat = self.v_delta[idx].clone() / (1.0 - self.y_v.powf(self.k as f64)); 
            let s_hat = self.s_delta[idx].clone() / (1.0 - self.y_s.powf(self.k as f64)); 
            let param_delta = self.alpha * v_hat / (s_hat.mapv(f64::sqrt) + self.epsilon); 

            model.graph.mut_node_output(param, parameter.output() - param_delta);  
        }
    }

    fn reset(&mut self, model: &mut SGD) {
        println!("Testing reset.."); 
    }

}
