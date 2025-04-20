use crate::graph::Dendrite;
//use crate::ops::{Add, Sub, Mul, Div}; 

/*
/// Unary operations for scalar values
pub trait UnaryOperation<T> {

    fn u_add(&mut self, rhs: T) -> &mut Dendrite<T>;

    fn u_sub(&mut self, rhs: T) -> &mut Dendrite<T>; 

    fn u_mul(&mut self, rhs: T) -> &mut Dendrite<T>; 

    fn u_div(&mut self, rhs: T) -> &mut Dendrite<T>; 
}


macro_rules! scalar_unary_ops {

    ($t:ident) => {

        impl UnaryOperation<$t> for Dendrite<$t> {

            fn u_add(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Box::new(Add)).unwrap()  
            }

            fn u_sub(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Box::new(Sub)).unwrap() 
            }

            fn u_mul(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Box::new(Mul)).unwrap() 
            }

            fn u_div(&mut self, rhs: $t) -> &mut Dendrite<$t> {
                self.unary(rhs, Box::new(Div)).unwrap() 
            }

        }

    }

}

scalar_unary_ops!(f32); 
scalar_unary_ops!(f64); 
scalar_unary_ops!(i32);
scalar_unary_ops!(i64);
scalar_unary_ops!(u8);
scalar_unary_ops!(u16); 
scalar_unary_ops!(usize);
*/
