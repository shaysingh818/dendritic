use ndarray::Array2;
use std::fmt::Debug; 

use crate::graph::*; 
use crate::ops::*; 


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


/// Shared trait for constructing scalar binary operations.
pub trait LossFunction<T> {

    fn mse(&mut self, val: T) -> &mut ComputationGraph<T>; 

}


impl LossFunction<Array2<f64>> for ComputationGraph<Array2<f64>> {

    fn mse(&mut self, val: Array2<f64>) -> &mut ComputationGraph<Array2<f64>> {
        self.unary(val, Box::new(MSE))
    }

}

