use std::fmt::Debug;
use std::collections::{HashMap, HashSet}; 

use ndarray::{arr2, Array2};
use log::{debug, warn, info}; 

use crate::node::{Node}; 
use crate::graph::ComputationGraph; 
use crate::operations::base::*; 
use crate::operations::activation::*; 
use crate::operations::arithmetic::*; 
use crate::operations::loss::*; 


pub trait DefaultOperations<T> {
    
    fn register_default_operations(&mut self);
}


macro_rules! default_ops {

    ($t:ty) => {

        impl DefaultOperations<$t> for ComputationGraph<$t> {

            fn register_default_operations(&mut self) {

                // default arithmetic operations
                self.register("Add", Box::new(Add)); 
                self.register("Mul", Box::new(Mul)); 
                self.register("Sub", Box::new(Sub)); 

                // default activation operations
                self.register("Tanh", Box::new(Tanh)); 
                self.register("Sigmoid", Box::new(Sigmoid)); 

                // default activation operations
                self.register("MSE", Box::new(MSE)); 
                self.register(
                    "BinaryCrossEntropy", 
                    Box::new(BinaryCrossEntropy)
                );
                self.register(
                    "DefaultLossFunction", 
                    Box::new(DefaultLossFunction)
                ); 

            }

        }

    }

}

default_ops!(f64); 
default_ops!(Array2<f64>); 

