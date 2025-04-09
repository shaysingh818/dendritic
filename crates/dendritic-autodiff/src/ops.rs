use std::fmt::Debug;
use std::cell::{RefCell, RefMut}; 
use crate::tensor::Tensor;
use crate::node::{Node};
//use ndarray::{arr2, Array2};


/// Structure for capturing binary and unary operations
#[derive(Debug)]
pub struct Add<LHS, RHS, T> {
    pub lhs: RefCell<LHS>,
    pub rhs: RefCell<RHS>,
    pub output: RefCell<T>,
    pub gradient: RefCell<T>
}

impl<RHS, LHS, T: Clone + Default> Add<RHS, LHS, T>
where
    RHS: Node<T>,
    LHS: Node<T>
{

    pub fn new(rhs: RHS, lhs: LHS) -> Add<LHS, RHS, T> {

       Add {
            lhs: RefCell::new(lhs),
            rhs: RefCell::new(rhs),
            output: RefCell::new(T::default()),
            gradient: RefCell::new(T::default())
       }
    }

    pub fn rhs(&self) -> RefMut<dyn Node<T>> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node<T>> {
        self.lhs.borrow_mut()
    }

}

macro_rules! scalar_add_op {

    ($t:ident) => {

        impl<RHS, LHS> Node<$t> for Add<RHS, LHS, $t> 
        where
            RHS: Node<$t>,
            LHS: Node<$t>
        {
            
            fn forward(&mut self) -> $t {
                let result = self.lhs().forward() + self.rhs().forward();
                self.output = RefCell::new(result.clone());
                result
            }

            fn backward(&mut self) {
                let result = self.lhs().forward() + self.rhs().forward();
                self.gradient = RefCell::new(result); 
            }

        }

    }
}

scalar_add_op!(i32); 
scalar_add_op!(i64); 
scalar_add_op!(f32); 
scalar_add_op!(f64); 
scalar_add_op!(usize);


/// Structure for capturing subtraction operation
#[derive(Debug)]
pub struct Sub<LHS, RHS, T> {
    pub lhs: RefCell<LHS>,
    pub rhs: RefCell<RHS>,
    pub output: RefCell<T>,
    pub gradient: RefCell<T>
}

impl<RHS, LHS, T: Clone + Default> Sub<RHS, LHS, T>
where
    RHS: Node<T>,
    LHS: Node<T>
{

    pub fn new(rhs: RHS, lhs: LHS) -> Sub<LHS, RHS, T> {

       Sub {
            lhs: RefCell::new(lhs),
            rhs: RefCell::new(rhs),
            output: RefCell::new(T::default()),
            gradient: RefCell::new(T::default())
       }
    }

    pub fn rhs(&self) -> RefMut<dyn Node<T>> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node<T>> {
        self.lhs.borrow_mut()
    }

}

macro_rules! scalar_subtraction_op {

    ($t:ident) => {

        impl<RHS, LHS> Node<$t> for Sub<RHS, LHS, $t> 
        where
            RHS: Node<$t>,
            LHS: Node<$t>
        {
            
            fn forward(&mut self) -> $t {
                let result = self.lhs().forward() - self.rhs().forward();
                self.output = RefCell::new(result.clone());
                result
            }

            fn backward(&mut self) {
                let result = self.lhs().forward() - self.rhs().forward();
                self.gradient = RefCell::new(result); 
            }

        }

    }
}

scalar_subtraction_op!(i32); 
scalar_subtraction_op!(i64); 
scalar_subtraction_op!(f32); 
scalar_subtraction_op!(f64); 
scalar_subtraction_op!(usize);


/// Structure for capturing subtraction operation
#[derive(Debug)]
pub struct Mul<LHS, RHS, T> {
    pub lhs: RefCell<LHS>,
    pub rhs: RefCell<RHS>,
    pub output: RefCell<T>,
    pub gradient: RefCell<T>
}

impl<RHS, LHS, T: Clone + Default> Mul<RHS, LHS, T>
where
    RHS: Node<T>,
    LHS: Node<T>
{

    pub fn new(rhs: RHS, lhs: LHS) -> Mul<LHS, RHS, T> {

       Mul {
            lhs: RefCell::new(lhs),
            rhs: RefCell::new(rhs),
            output: RefCell::new(T::default()),
            gradient: RefCell::new(T::default())
       }
    }

    pub fn rhs(&self) -> RefMut<dyn Node<T>> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node<T>> {
        self.lhs.borrow_mut()
    }

}

macro_rules! scalar_multiplication_op {

    ($t:ident) => {

        impl<RHS, LHS> Node<$t> for Mul<RHS, LHS, $t> 
        where
            RHS: Node<$t>,
            LHS: Node<$t>
        {
            
            fn forward(&mut self) -> $t {
                let result = self.lhs().forward() * self.rhs().forward();
                self.output = RefCell::new(result.clone());
                result
            }

            fn backward(&mut self) {
                let result = self.lhs().forward() * self.rhs().forward();
                self.gradient = RefCell::new(result); 
            }

        }

    }
}

scalar_multiplication_op!(i32); 
scalar_multiplication_op!(i64); 
scalar_multiplication_op!(f32); 
scalar_multiplication_op!(f64); 
scalar_multiplication_op!(usize); 


/// Structure for capturing subtraction operation
#[derive(Debug)]
pub struct Div<LHS, RHS, T> {
    pub lhs: RefCell<LHS>,
    pub rhs: RefCell<RHS>,
    pub output: RefCell<T>,
    pub gradient: RefCell<T>
}

impl<RHS, LHS, T: Clone + Default> Div<RHS, LHS, T>
where
    RHS: Node<T>,
    LHS: Node<T>
{

    pub fn new(rhs: RHS, lhs: LHS) -> Div<LHS, RHS, T> {

       Div {
            lhs: RefCell::new(lhs),
            rhs: RefCell::new(rhs),
            output: RefCell::new(T::default()),
            gradient: RefCell::new(T::default())
       }
    }

    pub fn rhs(&self) -> RefMut<dyn Node<T>> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node<T>> {
        self.lhs.borrow_mut()
    }

}

macro_rules! scalar_division_op {

    ($t:ident) => {

        impl<RHS, LHS> Node<$t> for Div<RHS, LHS, $t> 
        where
            RHS: Node<$t>,
            LHS: Node<$t>
        {
            
            fn forward(&mut self) -> $t {
                let result = self.lhs().forward() / self.rhs().forward();
                self.output = RefCell::new(result.clone());
                result
            }

            fn backward(&mut self) {
                let result = self.lhs().forward() / self.rhs().forward();
                self.gradient = RefCell::new(result); 
            }

        }

    }
}

scalar_division_op!(i32); 
scalar_division_op!(i64); 
scalar_division_op!(f32); 
scalar_division_op!(f64); 
scalar_division_op!(usize); 

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
