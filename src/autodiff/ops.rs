use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::Ops;
use crate::autodiff::node::{Node, Value}; 
use std::cell::{RefCell, RefMut}; 


pub struct Dot<RHS, LHS> {
    pub rhs: RefCell<RHS>,
    pub lhs: RefCell<LHS>,
    pub output: RefCell<Value<NDArray<f64>>>,
    pub gradient: RefCell<Value<NDArray<f64>>>
}


impl<RHS, LHS> Dot<RHS, LHS>
where
    RHS: Node,
    LHS: Node,
{

    pub fn new(rhs: RHS, lhs: LHS) -> Dot<RHS, LHS> {

        let op_result = rhs.value().dot(lhs.value().clone()).unwrap();
        let op_value = Value::new(&op_result);

        Dot {
            rhs: RefCell::new(rhs),
            lhs: RefCell::new(lhs),
            output: RefCell::new(op_value.clone()),
            gradient: RefCell::new(op_value)
        }
    }

    pub fn rhs(&self) -> RefMut<dyn Node> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node> {
        self.lhs.borrow_mut()
    }

}


impl <RHS, LHS>Node for Dot<RHS, LHS>
where
    RHS: Node,
    LHS: Node,    
{

    fn forward(&mut self) {

        let rhs = self.rhs().value();
        let lhs = self.lhs().value();

        self.rhs().forward();
        self.lhs().forward();

        let result = rhs.dot(lhs).unwrap();
        self.output = Value::new(&result).into(); 
    } 

    fn backward(&mut self, upstream_gradient: NDArray<f64>) {

        self.gradient = Value::new(&upstream_gradient).into();

        let rhs_t = self.rhs().value().transpose().unwrap();
        let lhs_t = self.lhs().value().transpose().unwrap();

        let rhs_grad = rhs_t.dot(upstream_gradient.clone()).unwrap();
        let lhs_grad = upstream_gradient.dot(lhs_t).unwrap();

        self.rhs().backward(lhs_grad);
        self.lhs().backward(rhs_grad);

    }


    fn value(&self) -> NDArray<f64> {
        self.output.borrow().val().clone()
    }
 
    fn grad(&self) -> NDArray<f64> {
        self.gradient.borrow().val().clone()
    }

    fn set_grad(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient = Value::new(&upstream_gradient).into();
    } 
}


pub struct ScaleAdd<RHS, LHS> 
where
    RHS: Node,
    LHS: Node,
{
    pub rhs: RefCell<RHS>,
    pub lhs: RefCell<LHS>,
    pub output: RefCell<Value<NDArray<f64>>>,
    pub gradient: RefCell<Value<NDArray<f64>>>
}



impl<RHS, LHS> ScaleAdd<RHS, LHS> 
where
    RHS: Node,
    LHS: Node,
{

    pub fn new(rhs: RHS, lhs: LHS) -> Self {

        let scalar_vec = lhs.value();
        let op_result = rhs.value().scale_add(scalar_vec).unwrap();
        let op_value = Value::new(&op_result);

        ScaleAdd {
            rhs: RefCell::new(rhs),
            lhs: RefCell::new(lhs),
            output: RefCell::new(op_value.clone()),
            gradient: RefCell::new(op_value)
        }
    }

    pub fn rhs(&self) -> RefMut<dyn Node> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node> {
        self.lhs.borrow_mut()
    }

}



impl<LHS, RHS> Node for ScaleAdd<RHS, LHS> 
where
    RHS: Node,
    LHS: Node,
{

    fn forward(&mut self) {

        self.rhs().forward();
        self.lhs().forward();

        let scalar_vec = self.lhs().value();
        let op_result = self.rhs().value().scale_add(scalar_vec).unwrap();
        self.output = Value::new(&op_result).into(); 
    } 

    fn backward(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient = Value::new(&upstream_gradient).into();
        self.lhs().backward(upstream_gradient.clone());
        self.rhs().backward(upstream_gradient);
    }

    fn value(&self) -> NDArray<f64> {
        self.output.borrow().val().clone()
    }

    fn grad(&self) -> NDArray<f64> {
        self.gradient.borrow().val().clone()
    }

    fn set_grad(&mut self, upstream_gradient: NDArray<f64>) {
        self.gradient = Value::new(&upstream_gradient).into();
    } 
}