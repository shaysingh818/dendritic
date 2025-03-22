use std::fmt::Debug;
use crate::tensor::Tensor;
use crate::node::{Operation}; 


#[derive(Debug, Clone)]
pub struct Add;

impl Operation<f64> for Add {

    fn forward(&self, inputs: &mut Vec<Tensor<f64>>) -> f64 {
        let lhs = inputs[0].value; 
        let rhs = inputs[1].value;
        lhs + rhs
    }

    fn backward(&self, inputs: &mut Vec<Tensor<f64>>) {
        for input in inputs {
            input.set_grad(1.0); 
        }
    }

}

#[derive(Debug, Clone)]
pub struct Sub; 

impl Operation<f64> for Sub {

    fn forward(&self, inputs: &mut Vec<Tensor<f64>>) -> f64 {
        let lhs = inputs[0].value; 
        let rhs = inputs[1].value;
        lhs - rhs
    }


    fn backward(&self, inputs: &mut Vec<Tensor<f64>>) {
        for input in inputs {
            input.set_grad(1.0); 
        }
    }

}


#[derive(Debug, Clone)]
pub struct Mul;

impl Operation<f64> for Mul {

    fn forward(&self, inputs: &mut Vec<Tensor<f64>>) -> f64 {
        let lhs = inputs[0].value; 
        let rhs = inputs[1].value;
        lhs * rhs
    }

    fn backward(&self, inputs: &mut Vec<Tensor<f64>>) {

        let mut lhs = inputs[0].clone();
        let mut rhs = inputs[1].clone();

        inputs[0].set_grad(*rhs.value()); 
        inputs[1].set_grad(*lhs.value()); 
    }

}


#[derive(Debug, Clone)]
pub struct Div;

impl Operation<f64> for Div {

    fn forward(&self, inputs: &mut Vec<Tensor<f64>>) -> f64 {
        let lhs = inputs[0].value; 
        let rhs = inputs[1].value;
        lhs / rhs
    }

    fn backward(&self, inputs: &mut Vec<Tensor<f64>>) {

        let mut lhs = inputs[0].clone();
        let mut rhs = inputs[1].clone();

        inputs[0].set_grad(1.0 / *rhs.value());
        inputs[1].set_grad(*lhs.value() / rhs.value().powf(2.0)); 
    }

}
