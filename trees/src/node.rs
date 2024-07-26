use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use metrics::utils::*;
use std::rc::Rc;
use std::cell::{RefCell, RefMut, Ref};

#[derive(Clone)]
pub struct Node {
    data: NDArray<f64>,
    threshold: f64,
    feature_idx: usize,
    value: Option<f64>,
    left: Option<NodeRef>,
    right: Option<NodeRef>
}

type NodeRef =  Rc<RefCell<Node>>;


impl Node {

    pub fn new(
        data: NDArray<f64>,
        threshold: f64,
        feature_idx: usize,
        left: Node,
        right: Node
    ) -> Node {

        Node {
           data: data, 
           threshold: threshold,
           feature_idx: feature_idx,
           value: None,
           left: Some(Rc::new(RefCell::new(left))),
           right: Some(Rc::new(RefCell::new(right)))
        }

    }

    pub fn leaf(value: f64) -> Node {

        Node {
           data: NDArray::new(vec![1,1]).unwrap(), 
           threshold: 0.0,
           feature_idx: 0,
           value: Some(value),
           left: None,
           right: None,
        }

    }


    pub fn data(&self) -> NDArray<f64> {
        self.data.clone()
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn feature_idx(&self) -> usize {
        self.feature_idx
    }

    pub fn value(&self) -> Option<f64> {
        match self.value {
            Some(value) => Some(value),
            None => None
        }
    }

    pub fn right(&self) -> Option<Node> {
        match &self.right {
            Some(right) => Some(right.borrow().clone()),
            None => None
        }
    }


    pub fn left(&self) -> Option<Node> {
        match &self.left {
            Some(left) => Some(left.borrow().clone()),
            None => None
        }
    }

}

