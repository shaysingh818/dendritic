use crate::graph::Dendrite;
use crate::ops::{Operation}; 

/// Unary operations for scalar values
pub trait UnaryOperation<T> {

    fn u_add(&mut self, rhs: T) -> &mut Dendrite<T>;

    fn u_sub(&mut self, rhs: T) -> &mut Dendrite<T>; 

    fn u_mul(&mut self, rhs: T) -> &mut Dendrite<T>; 

    fn u_div(&mut self, rhs: T) -> &mut Dendrite<T>; 
}


macro_rules! unary_methods {

    ($t:ident) => {

        impl UnaryOperation<$t> for Dendrite<$t> {

            fn u_add(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Operation::add())  
            }

            fn u_sub(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Operation::add()) 
            }

            fn u_mul(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Operation::add())
            }

            fn u_div(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Operation::add())
            }

        }

    }

}

//unary_methods!(f32); 
unary_methods!(f64); 
//unary_methods!(i32);
//unary_methods!(i64);
//unary_methods!(u8);
//unary_methods!(u16); 
//unary_methods!(usize);
