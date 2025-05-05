use std::fmt; 
use std::fmt::{Debug, Display};
use crate::tensor::Tensor;
use crate::node::{Node, Node2}; 
use crate::graph::ComputationGraph; 
use chrono::Local; 
use ndarray::{arr2, Array2};



/// General purpose logging function
pub fn debug_log(msg: &str) {
    #[cfg(debug_assertions)]
    {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        println!("[{}]: {}", log_time, msg);
    }
}


/// Base operation trait for allowing shared behavior for operations
#[derive(Clone, Debug)]
pub struct Operation<T> {

    /// method for defining forward pass behavior of operation
    pub forward: fn(nodes: &Vec<Node<T>>, curr_node_idx: usize) -> T,

    /// method for defining backward pass behavior of operation
    pub backward: fn(nodes: &mut Vec<Node<T>>, curr_node_idx: usize), 

}


pub trait OperationTrait<T>: OperationClone<T> + Debug {

    fn forward(
        &self, 
        nodes: &Vec<Node2<T>>, 
        curr_node_idx: usize) -> T;
 
    fn backward(
        &self, 
        nodes: &mut Vec<Node2<T>>, 
        curr_node_idx: usize);

}


pub trait OperationClone<T> {
    fn clone_box(&self) -> Box<dyn OperationTrait<T>>;
}

impl<T, U> OperationClone<T> for U
where
    U: 'static + OperationTrait<T> + Clone,
{
    fn clone_box(&self) -> Box<dyn OperationTrait<T>> {
        Box::new(self.clone())
    }
}


impl<T> Clone for Box<dyn OperationTrait<T>> {
    fn clone(&self) -> Box<dyn OperationTrait<T>> {
        self.clone_box()
    }
} 


/*
impl<T: Clone> Debug for Box<dyn OperationTrait<T>> {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "This is an operation trait") 
    }

} */


#[derive(Clone, Debug)]
pub struct DefaultValue; 


impl<T> OperationTrait<T> for DefaultValue 
where 
    T: Clone + Default, 
{

    fn forward(
        &self, 
        nodes: &Vec<Node2<T>>, 
        curr_node_idx: usize) -> T {

        nodes[curr_node_idx].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node2<T>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Backward on raw value idx: {}",
                curr_node_idx
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct Add; 


impl OperationTrait<f64> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node2<f64>>, 
        curr_node_idx: usize) -> f64 {

        debug_log(
            &format!(
                "Performing forward pass add on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward add upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        nodes[inputs[0]].output() + nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node2<f64>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward addition on node index: {}",
                curr_node_idx
            ) 
        ); 

        let node_inputs = nodes[curr_node_idx].inputs();
        for (idx, input) in node_inputs.iter().enumerate() {
            nodes[node_inputs[idx]].set_grad_output(1.0);
        }

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                node_inputs
            ) 
        ); 

    }
}



impl<T: Clone + Default> Operation<T> {

    pub fn default() -> Self {

        pub fn forward<T>(
            nodes: &Vec<Node<T>>, 
            curr_node_idx: usize) -> T where T: Clone + Default {
 
            nodes[curr_node_idx].output()
        }

        pub fn backward<T>(nodes: &mut Vec<Node<T>>, curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Backward on raw value idx: {}",
                    curr_node_idx
                ) 
            ); 

        }

        Operation {
            forward: forward,
            backward: backward, 
        }

    }

}


impl Operation<f64> {

    pub fn add() -> Self {

        pub fn forward(nodes: &Vec<Node<f64>>, curr_node_idx: usize) -> f64 {
            debug_log(
                &format!(
                    "Performing forward pass add on node index: {}",
                    curr_node_idx
                ) 
            ); 

            debug_log(
                &format!(
                    "Forward add upstream values: {:?}",
                    nodes[curr_node_idx].upstream()
                ) 
            );

            let node_inputs = nodes[curr_node_idx].inputs();
            nodes[node_inputs[0]].output() + nodes[node_inputs[1]].output()
        }

        pub fn backward(nodes: &mut Vec<Node<f64>>, curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Performing backward addition on node index: {}",
                    curr_node_idx
                ) 
            ); 

            let node_inputs = nodes[curr_node_idx].inputs();
            for (idx, input) in node_inputs.iter().enumerate() {
                nodes[node_inputs[idx]].set_grad_output(1.0);
            }

            debug_log(
                &format!(
                    "Updated gradients for node input indexes: {:?}",
                    node_inputs
                ) 
            ); 

        }

        Operation {
            forward: forward,
            backward: backward, 
        }
    }


    pub fn sub() -> Self {

        pub fn forward(nodes: &Vec<Node<f64>>, curr_node_idx: usize) -> f64 {

            debug_log(
                &format!(
                    "Performing forward pass subtraction: {}",
                    nodes[curr_node_idx].output()
                ) 
            ); 

            let node_inputs = nodes[curr_node_idx].inputs();
            nodes[node_inputs[0]].output() - nodes[node_inputs[1]].output()
        }

        pub fn backward(nodes: &mut Vec<Node<f64>>, curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Performing backward pass subtraction: {}",
                    nodes[curr_node_idx].output()
                ) 
            ); 

            let node_inputs = nodes[curr_node_idx].inputs();
            for (idx, input) in node_inputs.iter().enumerate() {
                nodes[node_inputs[idx]].set_grad_output(1.0);
            }
        }

        Operation {
            forward: forward,
            backward: backward, 
        }
    }


    // goes here
    pub fn mul() -> Self {

        pub fn forward(nodes: &Vec<Node<f64>>, curr_node_idx: usize) -> f64 {

            debug_log(
                &format!(
                    "Performing forward pass multiplication, current output: {}",
                    nodes[curr_node_idx].output()
                ) 
            ); 

            let node_inputs = nodes[curr_node_idx].inputs();
            nodes[node_inputs[0]].output() * nodes[node_inputs[1]].output()
        }

        pub fn backward(nodes: &mut Vec<Node<f64>>, curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Performing backward pass multiplication: {}",
                    nodes[curr_node_idx].output()
                ) 
            );

            let node_inputs = nodes[curr_node_idx].inputs();
            let lhs = nodes[node_inputs[0]].output(); 
            let rhs = nodes[node_inputs[1]].output(); 

            nodes[node_inputs[0]].set_grad_output(rhs); 
            nodes[node_inputs[1]].set_grad_output(lhs);
        }

        Operation {
            forward: forward,
            backward: backward, 
        }
    }

}

/*
impl Operation<Array2<f64>> {

    fn add() -> Self {

        pub fn forward(
            nodes: &Vec<Node<Array2<f64>>>, 
            curr_node_idx: usize) -> Array2<f64> {

            debug_log(
                &format!(
                    "Performing forward pass addition ndarray: {}",
                    curr_node_idx
                ) 
            );

            let inputs = nodes[curr_node_idx].inputs();
            nodes[inputs[0]].output() + nodes[inputs[1]].output()
        }


        pub fn backward(
            nodes: &mut Vec<Node<Array2<f64>>>, 
            curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Performing backward pass addition ndarray: {}",
                    curr_node_idx
                ) 
            );

            let node_upstream = nodes[curr_node_idx].upstream();
            let node_inputs = nodes[curr_node_idx].inputs();
    
        }

        Operation {
            forward: forward,
            backward: backward, 
        }

    }


    fn sub() -> Self {

        pub fn forward(
            nodes: &Vec<Node<Array2<f64>>>, 
            curr_node_idx: usize) -> Array2<f64> {

            debug_log(
                &format!(
                    "Performing forward pass addition ndarray: {}",
                    curr_node_idx
                ) 
            );

            let inputs = nodes[curr_node_idx].inputs();
            nodes[inputs[0]].output() - nodes[inputs[1]].output()
        }


        pub fn backward(
            nodes: &mut Vec<Node<Array2<f64>>>, 
            curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Performing backward pass addition ndarray: {}",
                    curr_node_idx
                ) 
            );

            let node_upstream = nodes[curr_node_idx].upstream();
            let node_inputs = nodes[curr_node_idx].inputs();
    
        }

        Operation {
            forward: forward,
            backward: backward, 
        }

    }


    fn mul() -> Self {

        pub fn forward(
            nodes: &Vec<Node<Array2<f64>>>, 
            curr_node_idx: usize) -> Array2<f64> {

            debug_log(
                &format!(
                    "Performing forward pass addition ndarray: {}",
                    curr_node_idx
                ) 
            );

            let inputs = nodes[curr_node_idx].inputs();
            nodes[inputs[0]].output() * nodes[inputs[1]].output()
        }


        pub fn backward(
            nodes: &mut Vec<Node<Array2<f64>>>, 
            curr_node_idx: usize) {

            debug_log(
                &format!(
                    "Performing backward pass addition ndarray: {}",
                    curr_node_idx
                ) 
            );

            let node_upstream = nodes[curr_node_idx].upstream();
            let node_inputs = nodes[curr_node_idx].inputs();
    
        }

        Operation {
            forward: forward,
            backward: backward, 
        }

    }

}  */ 

/*

/// Trait implementation for 2 dimensional ndarrays
impl Operation<Array2<f64>> for Add {

    fn forward(
        &self, 
        inputs: Vec<Tensor<Array2<f64>>>, 
        prev: Array2<f64>) -> Array2<f64> {

        match inputs.len() {

            2 => { // Binary operation
                inputs[0].value() + inputs[1].value()
            },
            1 => { // Unary
                prev + inputs[0].value() 
            },
            _ => panic!("FORWARD ERROR: {}", "Inputs to add op incorrect"), 
        }
    }


    fn backward(
        &self, 
        inputs: &mut Vec<Tensor<Array2<f64>>>,
        mut prev: &mut Node<Array2<f64>>,
        upstream: Array2<f64>) {

        match inputs.len() {

            2 => {
                inputs[0].set_grad(upstream.clone()); 
                inputs[1].set_grad(upstream); 
            },

            1 => {
                prev.mut_output().set_grad(upstream.clone()); 
                inputs[0].set_grad(upstream); 
            },

            _ => panic!("BACKWARD ERROR: {}", "Inputs to add operation incorrect"), 

        }

    }
 
}


impl Operation<Array2<f64>> for Mul {

    fn forward(
        &self, 
        inputs: Vec<Tensor<Array2<f64>>>, 
        mut prev: Array2<f64>) -> Array2<f64> {

        match inputs.len() {

            2 => { // Binary operation
                let lhs = inputs[0].value(); 
                let rhs = inputs[1].value();
                lhs.dot(&rhs)
            },
            1 => { // Unary
                let rhs = inputs[0].value(); 
                prev.dot(&rhs) 
            },
            _ => panic!("FORWARD ERROR: {}", "Inputs to mul op incorrect"), 
        }
    }


    fn backward(
        &self, 
        inputs: &mut Vec<Tensor<Array2<f64>>>, 
        prev: &mut Node<Array2<f64>>,
        upstream: Array2<f64>) {

        let lhs = inputs[0].value();
        let rhs = inputs[1].value();
        let upstream_clone = upstream.clone(); 

        let rhs_grad = upstream_clone.dot(&rhs.t());
        let lhs_grad = lhs.t().dot(&upstream_clone); 
        println!("UPSTREAM: {:?}", upstream_clone.shape()); 
        println!("RHS: {:?}", rhs.t().shape()); 
        println!("LHS: {:?}", lhs.t().shape()); 
        //inputs[0].set_grad(lhs_grad); 
        //inputs[1].set_grad(rhs_grad); 

    }
 
} */


/*
pub trait OperationTrait<T>: OperationClone<T> {

    fn forward(&self, nodes: &Vec<Node2<T>>, curr_node_idx: usize) -> T;
 
    fn backward(&self, nodes: &mut Vec<Node2<T>>, curr_node_idx: usize);

}


pub trait OperationClone<T> {
    fn clone_box(&self) -> Box<dyn OperationTrait<T>>;
}

impl<T, U> OperationClone<T> for U
where
    U: 'static + OperationTrait<T> + Clone,
{
    fn clone_box(&self) -> Box<dyn OperationTrait<T>> {
        Box::new(self.clone())
    }
}


impl<T> Clone for Box<dyn OperationTrait<T>> {
    fn clone(&self) -> Box<dyn OperationTrait<T>> {
        self.clone_box()
    }
}


#[derive(Clone)]
pub struct ValueNode; 

impl OperationTrait<f64> for ValueNode {


    fn forward(&self, nodes: &Vec<Node2<f64>>, curr_node_idx: usize) -> f64 {
        nodes[curr_node_idx].value.value()
    }
 
    fn backward(&self, nodes: &mut Vec<Node2<f64>>, curr_node_idx: usize) {
        debug_log(
            &format!(
                "Backward on raw value idx: {}",
                curr_node_idx
            ) 
        ); 
    }

}
*/
