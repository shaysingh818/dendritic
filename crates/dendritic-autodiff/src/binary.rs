use crate::graph::Dendrite;
use crate::ops::{Add}; 
use crate::tensor::Tensor; 
use std::collections::HashSet; 

/// Shared trait for constructing scalar binary operations.
/// Function calls take in 2 required inputs of the same type
pub trait BinaryOperation<T> {

    /// Construct add operation on graph with 2 inputs
    fn add(&mut self, lhs: T, rhs: T) -> &mut Dendrite<T>; 

    //fn sub(&mut self, lhs: T, rhs: T) -> &mut Dendrite<T>; 

    //fn mul(&mut self, lhs: T, rhs: T) -> &mut Dendrite<T>; 

    //fn div(&mut self, lhs: T, rhs: T) -> &mut Dendrite<T>; 
}

macro_rules! scalar_binary_ops {

    ($t:ident) => {

        impl BinaryOperation<$t> for Dendrite<$t> {

            fn add(&mut self, lhs: $t, rhs: $t) -> &mut Dendrite<$t> {

                let lhs_val = Tensor::new(&lhs); 
                let rhs_val = Tensor::new(&rhs);
                let op = Add::new(lhs_val.clone(), rhs_val.clone());

                self.binary(
                    Box::new(lhs_val), 
                    Box::new(rhs_val),
                    Box::new(op)
                )
            }

            /*
            fn sub(&mut self, lhs: $t, rhs: $t) -> &mut Dendrite<$t> {
                self.binary(lhs, rhs, Box::new(Sub)) 
            }

            fn mul(&mut self, lhs: $t, rhs: $t) -> &mut Dendrite<$t> {
                self.binary(lhs, rhs, Box::new(Mul)) 
            }

            fn div(&mut self, lhs: $t, rhs: $t) -> &mut Dendrite<$t> {
                self.binary(lhs, rhs, Box::new(Div)) 
            } */
        }

    }

}



//scalar_binary_ops!(i32); 
//scalar_binary_ops!(i64); 
//scalar_binary_ops!(f32); 
scalar_binary_ops!(f64);
//scalar_binary_ops!(usize); 

