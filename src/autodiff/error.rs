use std::fmt; 
use std::error; 


/// Custom error type for errors related to computation graph structure
#[derive(Debug)]
pub enum GraphError {
    UnaryOperation,
    BinaryOperation,
    NodeRelation
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {

            GraphError::UnaryOperation =>  {
                write!(f, "{}", "Unary operations must have a preceding binary operation with 2 inputs")
            },

            GraphError::BinaryOperation => {
                write!(f, "{}", "Binary operations must have atleast 2 inputs")
            },

            GraphError::NodeRelation =>  {
                write!(f, "{}", "Could not create node relationship, specified node does not exist")

            },
        }
    }
}



impl error::Error for GraphError {} 
