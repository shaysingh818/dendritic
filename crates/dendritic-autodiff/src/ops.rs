use std::fmt; 
use std::fmt::{Debug, Display};
use crate::tensor::Tensor;
use crate::node::{Node}; 
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

pub trait Operation<T>: OperationClone<T> + Debug {

    fn forward(&self, nodes: &Vec<Node<T>>, curr_idx: usize) -> T;
 
    fn backward(&self, nodes: &mut Vec<Node<T>>, curr_idx: usize);

}


pub trait OperationClone<T> {
    fn clone_box(&self) -> Box<dyn Operation<T>>;
}

impl<T, U> OperationClone<T> for U
where
    U: 'static + Operation<T> + Clone,
{
    fn clone_box(&self) -> Box<dyn Operation<T>> {
        Box::new(self.clone())
    }
}


impl<T> Clone for Box<dyn Operation<T>> {
    fn clone(&self) -> Box<dyn Operation<T>> {
        self.clone_box()
    }
} 


#[derive(Clone, Debug)]
pub struct DefaultValue; 


impl<T> Operation<T> for DefaultValue 
where 
    T: Clone + Default, 
{

    fn forward(
        &self, 
        nodes: &Vec<Node<T>>, 
        curr_idx: usize) -> T {

        nodes[curr_idx].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<T>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "Backward on raw value idx: {}",
                curr_idx
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct Add; 


impl Operation<f64> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug_log(
            &format!(
                "(ADD SCALAR) Performing forward pass on node index: {:?}",
                curr_idx
            ) 
        ); 

        debug_log(
            &format!(
                "(ADD SCALAR) Forward add upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() + nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {

        debug_log(
            &format!(
                "(ADD SCALAR) Performing backward on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        let node_inputs = nodes[curr_idx].inputs();
        println!("ADD DEBUG: {:?}", node_inputs); 
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


impl Operation<Array2<f64>> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "(ADD) Performing forward pass on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "(ADD) Forward upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() + nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "(ADD) Performing backward on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let upstream = nodes[curr_idx].upstream(); 

        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();

        match upstream.len() {
            1 => {
                let upstream_grad = nodes[upstream[0]].grad(); 
                nodes[curr_idx].set_grad_output(upstream_grad);
            },
            0 => {
                panic!("No upstream values associated with node: {:?}", nodes[curr_idx]); 
            },
            _ => {
                panic!("ADD: Unable to handle upstream values");
            }
        }

        debug_log(
            &format!(
                "(ADD) Updated gradients for node input indexes: {:?}",
                inputs
            ) 
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

        debug_log(
            &format!(
                "Performing forward pass subtract on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "Forward subtraction upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() - nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {

        debug_log(
            &format!(
                "Performing backward subtract on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        let node_inputs = nodes[curr_idx].inputs();
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


impl Operation<Array2<f64>> for Sub {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward pass multiply on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "Forward multiply upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();
        lhs - rhs
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        debug_log(
            &format!(
                "Performing backward multiply on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
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

        nodes[node_inputs[0]].set_grad_output(rhs_grad); 
        nodes[node_inputs[1]].set_grad_output(lhs_grad);

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                node_inputs
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct Mul; 

impl Operation<f64> for Mul {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_idx: usize) -> f64 {

        debug_log(
            &format!(
                "(MUL) Performing forward pass on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "(MUL) Forward multiply upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        nodes[inputs[0]].output() * nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_idx: usize) {


        debug_log(
            &format!(
                "(MUL) Performing backward pass on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        let node_inputs = nodes[curr_idx].inputs();
        let lhs = nodes[node_inputs[0]].output(); 
        let rhs = nodes[node_inputs[1]].output();

        nodes[node_inputs[0]].set_grad_output(rhs); 
        nodes[node_inputs[1]].set_grad_output(lhs);

        debug_log(
            &format!(
                "(MUL) Updated gradients for node input indexes: {:?}",
                node_inputs
            ) 
        ); 

    }
}


impl Operation<Array2<f64>> for Mul {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "(MUL) Performing forward pass on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "(MUL) Forward multiply upstream values: {:?}",
                nodes[curr_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output(); 
        lhs.dot(&rhs)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_idx: usize) {

        debug_log(
            &format!(
                "(MUL) Performing backward multiply on node index: {:?}",
                nodes[curr_idx].inputs()
            ) 
        ); 

        let inputs = nodes[curr_idx].inputs();
        let upstream = nodes[curr_idx].upstream();

        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();

        if upstream.len() > 1 {
            panic!("Backward multiply can only handle one upstream"); 
        }

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

        debug_log(
            &format!(
                "(MUL) Updated gradients for node input indexes: {:?}",
                inputs
            ) 
        ); 

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

        debug_log(
            &format!(
                "Forward MSE upstream values: {:?}",
                nodes[curr_idx].upstream()
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
                nodes[curr_idx].inputs()
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
                nodes[curr_idx].inputs()
            ) 
        ); 

        debug_log(
            &format!(
                "Forward MSE upstream values: {:?}",
                nodes[curr_idx].upstream()
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
                "Performing backward multiply on node index: {:?}",
                nodes[curr_idx].inputs()
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
                "Performing backward multiply on node index: {:?}",
                nodes[curr_idx].inputs()
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
pub struct Sigmoid;

impl Operation<Array2<f64>> for Sigmoid {

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


        debug_log(
            &format!(
                "Performing backward multiply on node index: {:?}",
                nodes[curr_idx].inputs()
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
                nodes[inputs[0]].set_grad_output(grad); 

            },
            _ => {
                panic!("Sigmoid must only have 1 input"); 
            }
        }


        debug_log(
            &format!(
                "Updated gradients for sigmoid operation: {:?}",
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


