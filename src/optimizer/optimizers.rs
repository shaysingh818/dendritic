use ndarray::{Array2};

use crate::autodiff::node::*; 
use crate::optimizer::regression::sgd::*;
use crate::optimizer::model::*;


pub trait Optimizer {

    /// Parmeter update method for optimizers
    fn step<M: Model>(&mut self, model: &mut M);

}

pub struct DefaultOptimizer {

    /// Learning rate associated with model
    pub alpha: f64

}

impl Optimizer for DefaultOptimizer {

    fn step<M: Model>(&mut self, model: &mut M) {
        let params = model.graph().parameters();
        for (_idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph().node(param);
            let grad = parameter.grad() * self.alpha;
            let delta = parameter.output() - grad;
            model.update_parameter(param, delta);
        }
    }

}

pub struct Nesterov {

    /// Learning rate associated with model
    pub alpha: f64,
    
    /// Momentum factor associated with model
    pub beta: f64,

    /// Velocity associated with paramters
    pub v: Vec<Array2<f64>>

}

impl Nesterov {

    /// Nesterov optimization technique with default parameters initialized
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

    fn step<M: Model>(&mut self, model: &mut M) {
        let params = model.graph().parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph().node(param);
            let lookahead = parameter.output() - (self.beta * self.v[idx].clone());
            model.update_parameter(param, lookahead);

            let grad = parameter.grad() * self.alpha;
            self.v[idx] = grad + (self.beta * self.v[idx].clone());
            let new_param = parameter.output() - self.v[idx].clone();
            model.update_parameter(param, new_param);
        }
    }

}


pub struct Adagrad {

    /// Learning rate associated with model
    pub alpha: f64,
    
    /// Momentum factor associated with model
    pub epsilon: f64,

    /// Not known yet
    pub s: Vec<Array2<f64>>

}

impl Adagrad {

    /// Default adagrad constructor using SGD
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

    fn step<M: Model>(&mut self, model: &mut M) {
        let params = model.graph().parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph().node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);
            self.s[idx] += &grad_squared;

            let ada = self.alpha / (self.s[idx].mapv(f64::sqrt) + self.epsilon);
            let param_new = parameter.output() - (ada * grad); 
            model.update_parameter(param, param_new);

        }
    }

}


pub struct RMSProp {

    /// Learning rate associated with model
    pub alpha: f64,
    
    /// Momentum factor associated with model
    pub epsilon: f64,

    /// Not known yet
    pub decay_rate: f64,

    /// Not known yet
    pub s: Vec<Array2<f64>>

}


impl RMSProp {

    /// Default RMSProp method using gradient descent
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

    fn step<M: Model>(&mut self, model: &mut M) {
        let params = model.graph().parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph().node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);

            self.s[idx] = self.decay_rate * self.s[idx].clone() + (1.0 - self.decay_rate) * grad_squared;
            let rms = self.alpha / (self.s[idx].mapv(f64::sqrt) + self.epsilon);
            let new = parameter.output() - (rms * grad); 
            model.update_parameter(param, new);
        }
    }

}


pub struct Adadelta {

    /// Learning rate associated with model
    pub y_s: f64,
   
    /// Learning rate associated with model
    pub y_x: f64,

    /// Momentum factor associated with model
    pub epsilon: f64,

    /// Not known yet
    pub s: Vec<Array2<f64>>,

    /// Not known yet
    pub u: Vec<Array2<f64>>

}


impl Adadelta {


    /// Default Adadelta method using gradient descent
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

    fn step<M: Model>(&mut self, model: &mut M) {
        let params = model.graph().parameters();
        for (idx, param) in params.into_iter().enumerate() {
            let parameter = model.graph().node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);
            
            self.s[idx] = self.y_s * self.s[idx].clone() + (1.0 - self.y_s) * grad_squared;
            let delta = ((self.u[idx].mapv(f64::sqrt) + self.epsilon) / (self.s[idx].mapv(f64::sqrt) + self.epsilon)) * grad;
            self.u[idx] = self.y_x * self.u[idx].clone() + (1.0 - self.y_x) * delta.mapv(|x| x * x); 
            let new = parameter.output() + (delta * -1.0);
            model.update_parameter(param, new); 
        }
    }
}


pub struct Adam {

    /// Learning rate associated with model
    pub alpha: f64,

    /// Epsilon to represent small value
    pub epsilon: f64, 

    /// First gradient decay
    pub y_v: f64,

    /// Second gradient decay
    pub y_s: f64,

    /// Step counter (iteration count)
    pub k: usize,

    /// First momentum estimation
    pub v_delta: Vec<Array2<f64>>,

    /// Second momentum estimation
    pub s_delta: Vec<Array2<f64>>
    
}


impl Adam {

    /// Default Adam optimizer using gradient descent
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

    /// Retrieve first gradient decay variable
    pub fn grad_decay_1(&self) -> f64 {
        self.y_v
    }
        
    /// Retrieve second gradient decay variable
    pub fn grad_decay_2(&self) -> f64 {
        self.y_s
    }

    /// Retrieve step count iteration of training
    pub fn step_count(&self) -> usize {
        self.k
    }

    /// Retrieve first set of momentum variables
    pub fn first_momentum(&self) -> Vec<Array2<f64>> {
        self.v_delta.clone()
    }

    /// Retrieve second set of momentum variables
    pub fn second_momentum(&self) -> Vec<Array2<f64>> {
        self.v_delta.clone()
    }

    /// Initiate zero set of momentum parameters based on shapes of marked parmeters
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

    fn step<M: Model>(&mut self, model: &mut M) {
        let params = model.graph().parameters();
        for (idx, param) in params.into_iter().enumerate() {

            let parameter = model.graph().node(param);
            let grad = parameter.grad();
            let grad_squared = grad.mapv(|x| x * x);

            self.v_delta[idx] = self.y_v * self.v_delta[idx].clone() + (1.0 - self.y_v) * grad.clone(); 
            self.s_delta[idx] = self.y_s * self.s_delta[idx].clone() + (1.0 - self.y_s) * grad_squared;

            if idx == 0 {
                self.k += 1;
            }

            let v_hat = self.v_delta[idx].clone() / (1.0 - self.y_v.powf(self.k as f64)); 
            let s_hat = self.s_delta[idx].clone() / (1.0 - self.y_s.powf(self.k as f64)); 
            let param_delta = self.alpha * v_hat / (s_hat.mapv(f64::sqrt) + self.epsilon); 
            model.update_parameter(param, parameter.output() - param_delta);  
        }
    }

}

