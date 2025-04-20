use std::fmt; 
use std::fmt::Debug;
use crate::tensor::Tensor;
use crate::node::Node; 
use crate::graph::Dendrite; 
//use ndarray::{arr2, Array2};


/// Base operation trait for allowing shared behavior for operations
#[derive(Clone, Debug)]
pub struct Operation<T> {

    /// method for defining forward pass behavior of operation
    pub forward: fn(nodes: &Vec<Node<T>>, input_idx: Vec<usize>) -> T,

    /// method for defining backward pass behavior of operation
    pub backward: fn(nodes: Vec<Node<T>>), 
}



impl Operation<f64> {

    pub fn add() -> Self {

        pub fn forward(nodes: &Vec<Node<f64>>, inputs: Vec<usize>) -> f64 {
            println!("Doing add forward pass");
            nodes[inputs[0]].output() + nodes[inputs[1]].output()
        }

        pub fn backward(nodes: Vec<Node<f64>>) {
            println!("Nothing for now"); 
        }

        Operation {
            forward: forward,
            backward: backward, 
        }
    }


}



/// Structure for capturing binary and unary operations
#[derive(Debug)]
pub struct Add; 

#[derive(Debug)]
pub struct Sub; 

#[derive(Debug)]
pub struct Mul; 

#[derive(Debug)]
pub struct Div;

/*
impl Operation<f64> for Add {

    fn forward(&self, graph: Dendrite<f64>, input_idx: Vec<usize>) -> f64 {
        if input_idx.len() != 2 {
            panic!("Add operation must have 2 inputs"); 
        }

        let lhs = graph.node(input_idx[0]); 
        let rhs = graph.node(input_idx[1]); 
        lhs.output() + rhs.output()
    }

    fn backward(&self, graph: Dendrite<f64>) {
        println!("Doing nothing"); 
    }


} */ 




/*
macro_rules! scalar_add_op {

    ($t:ident) => {

        impl Operation<$t> for Add {
            
            fn forward(
                &self, 
                inputs: Vec<Tensor<$t>>, 
                prev: Tensor<$t>) -> $t {
                
                match inputs.len() {
                    2 => { // binary 
                        inputs[0].value() + inputs[1].value()
                    },
                    1 => { // unary
                        inputs[0].value() + prev.value()
                    },
                    _ => {
                        panic!("Unknown amount of inputs for add"); 
                    }
                }
            }

            fn backward(
                &self,
                inputs: &mut Vec<Tensor<$t>>,
                prev: &mut Node<$t>) {

                for input in inputs {
                    input.set_grad(1.0 as $t);
                }
            }

        }

        impl Operation<$t> for Sub {
            
            fn forward(
                &self, 
                inputs: Vec<Tensor<$t>>, 
                prev: Tensor<$t>) -> $t {
                
                match inputs.len() {
                    2 => { // binary 
                        inputs[0].value() - inputs[1].value()
                    },
                    1 => { // unary
                        inputs[0].value() - prev.value() 
                    },
                    _ => {
                        panic!("Unknown amount of inputs for add"); 
                    }
                }
            }

            fn backward(
                &self,
                inputs: &mut Vec<Tensor<$t>>,
                prev: &mut Node<$t>) {

                for input in inputs {
                    input.set_grad(1.0 as $t);
                }
            }

        }


        impl Operation<$t> for Mul {
            
            fn forward(
                &self, 
                inputs: Vec<Tensor<$t>>, 
                prev: Tensor<$t>) -> $t {
                
                match inputs.len() {
                    2 => { // binary 
                        inputs[0].value() * inputs[1].value()
                    },
                    1 => { // unary
                        inputs[0].value() * prev.value() 
                    },
                    _ => {
                        panic!("Unknown amount of inputs for add"); 
                    }
                }
            }

            fn backward(
                &self,
                inputs: &mut Vec<Tensor<$t>>,
                prev: &mut Node<$t>) {

                match inputs.len() {
                    2 => { // binary 
                    
                        let mut lhs = inputs[0].clone(); 
                        let mut rhs = inputs[1].clone(); 

                        inputs[0].set_grad(rhs.value());
                        inputs[1].set_grad(lhs.value());
                        println!("BINARY MUL"); 
                    },
                    1 => { // unary
                        let mut lhs = prev.output().clone(); 
                        let mut rhs = inputs[0].clone();

                        inputs[0].set_grad(lhs.value());
                        prev.set_grad_output(rhs.value()); 

                    },
                    _ => {
                        panic!("Unknown amount of inputs for add"); 
                    }
                }

            }

        }


        impl Operation<$t> for Div {
            
            fn forward(
                &self, 
                inputs: Vec<Tensor<$t>>, 
                prev: Tensor<$t>) -> $t {
                
                match inputs.len() {
                    2 => { // binary 
                        inputs[0].value() / inputs[1].value()
                    },
                    1 => { // unary
                        inputs[0].value() / prev.value() 
                    },
                    _ => {
                        panic!("Unknown amount of inputs for add"); 
                    }
                }
            }

            fn backward(
                &self,
                inputs: &mut Vec<Tensor<$t>>,
                prev: &mut Node<$t>) {

                match inputs.len() {
                    2 => { // binary 
                    
                        let mut lhs = inputs[0].clone(); 
                        let mut rhs = inputs[1].clone(); 

                        inputs[0].set_grad(lhs.value());
                        inputs[1].set_grad(rhs.value());
                    },
                    1 => { // unary
                        let mut lhs = prev.output().clone(); 
                        let mut rhs = inputs[0].clone();

                        inputs[0].set_grad(lhs.value());
                        prev.output().set_grad(rhs.value()); 

                    },
                    _ => {
                        panic!("Unknown amount of inputs for add"); 
                    }
                }


            }

        }

    }
}


scalar_add_op!(i32); 
scalar_add_op!(i64); 
scalar_add_op!(f32); 
scalar_add_op!(f64);
scalar_add_op!(u8);
scalar_add_op!(u16);
scalar_add_op!(usize);
*/

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
