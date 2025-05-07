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

    fn forward(
        &self, 
        nodes: &Vec<Node<T>>, 
        curr_node_idx: usize) -> T;
 
    fn backward(
        &self, 
        nodes: &mut Vec<Node<T>>, 
        curr_node_idx: usize);

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
        curr_node_idx: usize) -> T {

        nodes[curr_node_idx].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<T>>, 
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


impl Operation<f64> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node<f64>>, 
        curr_node_idx: usize) -> f64 {

        debug_log(
            &format!(
                "(ADD SCALAR) Performing forward pass on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "(ADD SCALAR) Forward add upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        nodes[inputs[0]].output() + nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "(ADD SCALAR) Performing backward on node index: {}",
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


impl Operation<Array2<f64>> for Add {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "(ADD) Performing forward pass on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "(ADD) Forward upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        nodes[inputs[0]].output() + nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "(ADD) Performing backward on node index: {}",
                curr_node_idx
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let upstream = nodes[curr_node_idx].upstream(); 

        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();

        if upstream.len() > 1 {
            panic!("Backward add can only handle one upstream"); 
        }

        let upstream_grad = nodes[upstream[0]].grad(); 
        nodes[curr_node_idx].set_grad_output(upstream_grad); 
        
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
        curr_node_idx: usize) -> f64 {

        debug_log(
            &format!(
                "Performing forward pass subtract on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward subtraction upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        nodes[inputs[0]].output() - nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward subtract on node index: {}",
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


impl Operation<Array2<f64>> for Sub {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward pass multiply on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward multiply upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();
        lhs - rhs
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        ); 

        let node_inputs = nodes[curr_node_idx].inputs();
        let node_upstream = nodes[curr_node_idx].upstream(); 

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
        curr_node_idx: usize) -> f64 {

        debug_log(
            &format!(
                "(MUL) Performing forward pass on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "(MUL) Forward multiply upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        nodes[inputs[0]].output() * nodes[inputs[1]].output()
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "(MUL) Performing backward pass on node index: {}",
                curr_node_idx
            ) 
        ); 

        let node_inputs = nodes[curr_node_idx].inputs();
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
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "(MUL) Performing forward pass on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "(MUL) Forward multiply upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output(); 
        lhs.dot(&rhs)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "(MUL) Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        ); 

        let inputs = nodes[curr_node_idx].inputs();
        let upstream = nodes[curr_node_idx].upstream(); 

        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output();

        if upstream.len() > 1 {
            panic!("Backward multiply can only handle one upstream"); 
        }

        let upstream = nodes[upstream[0]].grad();  
        let rhs_grad = upstream.dot(&rhs.t());
        let lhs_grad = lhs.t().dot(&upstream); 

        nodes[inputs[0]].set_grad_output(rhs_grad); 
        nodes[inputs[1]].set_grad_output(lhs_grad);

        debug_log(
            &format!(
                "(MUL) Updated gradients for node input indexes: {:?}",
                inputs
            ) 
        ); 

    }
}


#[derive(Clone, Debug)]
pub struct Regularization;

impl Operation<Array2<f64>> for Regularization {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward regularization on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward regularization upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output(); 

        let rhs_square = rhs.mapv(|x| x * x);
        //let rhs_sum = rhs_square.sum(); 
        lhs.dot(&rhs_square)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        ); 

        let node_inputs = nodes[curr_node_idx].inputs();
        let node_upstream = nodes[curr_node_idx].upstream(); 

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
pub struct L2Regularization;

impl Operation<Array2<f64>> for L2Regularization {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward regularization on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward regularization upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output(); 

        let rhs_square = rhs.mapv(|x| x * x);
        //let rhs_sum = rhs_square.sum(); 
        lhs.dot(&rhs_square)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        ); 

        let node_inputs = nodes[curr_node_idx].inputs();
        let node_upstream = nodes[curr_node_idx].upstream(); 

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
pub struct L1Regularization;

impl Operation<Array2<f64>> for L1Regularization {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward regularization on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward regularization upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let lhs = nodes[inputs[0]].output(); 
        let rhs = nodes[inputs[1]].output(); 

        let rhs_square = rhs.mapv(|x| x * x);
        //let rhs_sum = rhs_square.sum(); 
        lhs.dot(&rhs_square)
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        ); 

        let node_inputs = nodes[curr_node_idx].inputs();
        let node_upstream = nodes[curr_node_idx].upstream(); 

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
pub struct MSE;

impl Operation<Array2<f64>> for MSE {

    fn forward(
        &self, 
        nodes: &Vec<Node<Array2<f64>>>, 
        curr_node_idx: usize) -> Array2<f64> {

        debug_log(
            &format!(
                "Performing forward MSE on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward MSE upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
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
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_node_idx].set_grad_output(grad); 

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
        curr_node_idx: usize) -> f64 {

        debug_log(
            &format!(
                "Performing forward MSE on node index: {}",
                curr_node_idx
            ) 
        ); 

        debug_log(
            &format!(
                "Forward MSE upstream values: {:?}",
                nodes[curr_node_idx].upstream()
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();

        let diff = y_true.clone() - y_pred;
        diff.powf(2.0) 
    }

    fn backward(
        &self, 
        nodes: &mut Vec<Node<f64>>, 
        curr_node_idx: usize) {


        debug_log(
            &format!(
                "Performing backward multiply on node index: {}",
                curr_node_idx
            ) 
        );

        let inputs = nodes[curr_node_idx].inputs();
        let y_pred = nodes[inputs[0]].output(); 
        let y_true = nodes[inputs[1]].output();
        let grad = y_pred - y_true;
        nodes[curr_node_idx].set_grad_output(grad); 

        debug_log(
            &format!(
                "Updated gradients for node input indexes: {:?}",
                inputs
            ) 
        ); 

    }
}
