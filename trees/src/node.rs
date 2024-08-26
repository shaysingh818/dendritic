use ndarray::ndarray::NDArray;
use std::rc::Rc;
use std::cell::{RefCell};
use serde::{Serialize, Deserialize}; 


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeSerialized {
    threshold: f64,
    feature_idx: usize,
    value: Option<f64>,
    mse: Option<f64>,
    information_gain: Option<f64>,
    pub left: Option<Box<NodeSerialized>>,
    pub right: Option<Box<NodeSerialized>>,
}


#[derive(Clone)]
pub struct Node {
    data: NDArray<f64>,
    threshold: f64,
    feature_idx: usize,
    value: Option<f64>,
    mse: Option<f64>,
    information_gain: Option<f64>,
    pub left: Option<NodeRef>,
    pub right: Option<NodeRef>
}


type NodeRef =  Rc<RefCell<Node>>;


impl Node {

    pub fn new(
        data: NDArray<f64>,
        threshold: f64,
        feature_idx: usize,
        information_gain: f64,
        left: Node,
        right: Node
    ) -> Node {

        Node {
           data: data, 
           threshold: threshold,
           feature_idx: feature_idx,
           value: None,
           mse: None,
           information_gain: Some(information_gain),
           left: Some(Rc::new(RefCell::new(left))),
           right: Some(Rc::new(RefCell::new(right)))
        }

    }


    pub fn regression(
        data: NDArray<f64>,
        threshold: f64,
        feature_idx: usize,
        mse: f64,
        left: Node,
        right: Node
    ) -> Node {

        Node {
           data: data, 
           threshold: threshold,
           feature_idx: feature_idx,
           value: None,
           mse: Some(mse),
           information_gain: None,
           left: Some(Rc::new(RefCell::new(left))),
           right: Some(Rc::new(RefCell::new(right)))
        }

    }

    pub fn leaf(value: f64) -> Node {

        Node {
           data: NDArray::new(vec![1,1]).unwrap(), 
           threshold: 0.0,
           feature_idx: 0,
           information_gain: None,
           mse: None,
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

    pub fn information_gain(&self) -> Option<f64> {
        match self.information_gain {
            Some(value) => Some(value),
            None => None
        }
    }

    pub fn mse(&self) -> Option<f64> {
        match self.mse {
            Some(value) => Some(value),
            None => None
        }
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

    pub fn save(&self) -> NodeSerialized {
        
        let node = NodeSerialized {
            threshold: self.threshold(),
            value: self.value(),
            feature_idx: self.feature_idx(),
            mse: self.mse(),
            information_gain: self.information_gain(),
            left: None,
            right: None
        };

        node
    }

    pub fn load(node: NodeSerialized) -> Node {

        Node {
           data: NDArray::new(vec![1,1]).unwrap(), 
           threshold: node.threshold,
           feature_idx: node.feature_idx,
           value: node.value,
           mse: node.mse,
           information_gain: node.information_gain,
           left: None,
           right: None
        }

    }


}

