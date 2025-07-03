use ndarray::Array2;

use crate::graph::ComputationGraph; 
use crate::operations::base::*; 
use crate::operations::activation::*; 
use crate::operations::arithmetic::*; 
use crate::operations::loss::*; 


pub trait DefaultOperations<T> {
    
    fn register_default_operations(&mut self);
}

/// Registry of supported operations by default. 
/// Operation registry for computation graph can be extended using this trait pattern implementation
macro_rules! default_ops {

    ($t:ty) => {

        impl DefaultOperations<$t> for ComputationGraph<$t> {

            fn register_default_operations(&mut self) {

                // default value
                self.register("DefaultValue", Box::new(DefaultValue)); 

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

