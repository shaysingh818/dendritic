use ndarray::ndarray::NDArray;
use ndarray::ops::Ops;
use std::cell::{RefCell, RefMut}; 
use crate::node::{Node, Value}; 

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

        self.rhs().backward(rhs_grad);
        self.lhs().backward(lhs_grad);

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


pub struct Regularization<RHS, LHS> 
where
    RHS: Node,
    LHS: Node,
{
    pub rhs: RefCell<RHS>,
    pub lhs: RefCell<LHS>,
    pub output: RefCell<Value<NDArray<f64>>>,
    pub gradient: RefCell<Value<NDArray<f64>>>,
    pub learning_rate: f64
}



impl<RHS, LHS> Regularization<RHS, LHS> 
where
    RHS: Node,
    LHS: Node,
{

    pub fn new(rhs: RHS, lhs: LHS, learning_rate: f64) -> Self {

        let weights = rhs.value();
        let w_square = weights.square().unwrap();
        let w_sum = w_square.sum().unwrap();
        let op_result = lhs.value().mult(w_sum).unwrap();
        let op_value = Value::new(&op_result);

        Regularization {
            rhs: RefCell::new(rhs),
            lhs: RefCell::new(lhs),
            output: RefCell::new(op_value.clone()),
            gradient: RefCell::new(op_value),
            learning_rate: learning_rate
        }
    }

    pub fn rhs(&self) -> RefMut<dyn Node> {
        self.rhs.borrow_mut()
    }

    pub fn lhs(&self) -> RefMut<dyn Node> {
        self.lhs.borrow_mut()
    }

}


impl<LHS, RHS> Node for Regularization<RHS, LHS> 
where
    RHS: Node,
    LHS: Node,
{

    fn forward(&mut self) {

        self.rhs().forward();
        self.lhs().forward();

        let weights = self.rhs().value();
        let w_square = weights.square().unwrap();
        let w_sum = w_square.sum().unwrap();
        let op_result = self.lhs.borrow().value().mult(w_sum).unwrap();
        self.output = Value::new(&op_result).into(); 
    } 

    fn backward(&mut self, upstream_gradient: NDArray<f64>) {
        let lr = self.learning_rate / upstream_gradient.size() as f64;
        let alpha = self.lhs().value().scalar_mult(2.0 * lr).unwrap();
        let weight_update = self.rhs().value().scale_mult(alpha).unwrap();
        self.gradient = Value::new(&weight_update).into();
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

