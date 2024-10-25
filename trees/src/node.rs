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

    /// Create new instance of decision tree classifier node
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


    /// Create new instance of decision tree regressor node
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


    /// Create new instance of decision tree leaf node
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

    /// Retrieve data attribute of decision node
    pub fn data(&self) -> NDArray<f64> {
        self.data.clone()
    }

    /// Retrieve threshold attribute of decision node
    pub fn threshold(&self) -> f64 {
        self.threshold
    }


    /// Retrieve feature index attribute of decision node
    pub fn feature_idx(&self) -> usize {
        self.feature_idx
    }


    /// Retrieve information gain attribute of decision node
    pub fn information_gain(&self) -> Option<f64> {
        match self.information_gain {
            Some(value) => Some(value),
            None => None
        }
    }

    /// Get mean squared error attribute of decision node
    pub fn mse(&self) -> Option<f64> {
        match self.mse {
            Some(value) => Some(value),
            None => None
        }
    }

    /// Retrieve value attribute of decision node
    pub fn value(&self) -> Option<f64> {
        match self.value {
            Some(value) => Some(value),
            None => None
        }
    }

    /// Retrieve right pointer of decision node
    pub fn right(&self) -> Option<Node> {
        match &self.right {
            Some(right) => Some(right.borrow().clone()),
            None => None
        }
    }

    /// Retrieve left pointer of decision node
    pub fn left(&self) -> Option<Node> {
        match &self.left {
            Some(left) => Some(left.borrow().clone()),
            None => None
        }
    }

    /// Save and serialize instance of decision node to JSON format
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

    /// Load instance of decision node to JSON format
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

