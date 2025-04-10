use std::collections::{HashMap, HashSet}; 
use std::rc::Rc;
use std::cell::{RefCell, RefMut, Ref}; 
use crate::node::{Node};
use crate::error::{GraphError};
use crate::tensor::{Tensor}; 

/// A dendrite is an instance of expression stored in a computation graph.
/// A dendrite stores an adjacency list of nodes (operations) in the
/// computation graph. Since nodes are not clonable and only references
/// of nodes can be used, the array of nodes is stored as smart pointers
/// that allow for interior mutability. 
#[derive(Debug)]
pub struct Dendrite<T> {

    /// references to node operations in the graph
    pub nodes: Vec<RefCell<Box<dyn Node<T>>>>,

    /// Adjacency list of indices referencing node array on graph
    pub adj_list: HashMap<usize, HashSet<usize>>,

    /// Temporary list to store path traversals in graph
    pub path: Vec<usize>,

    /// Index of current node ready for computation
    pub current_node_idx: usize,

    /// Index of node with the most recent computation 
    pub prev_node_idx: usize
}

impl<T: Clone> Dendrite<T> {

    /// Create new instance of computation graph structure
    pub fn new() -> Self {

        Dendrite {
            nodes: vec![],
            adj_list: HashMap::new(),
            path: vec![],
            current_node_idx: 0,
            prev_node_idx: 0
        }
    }


    /// Get current node in graph next in line for computation
    pub fn current_node_idx(&self) -> usize {
        self.current_node_idx 
    }

    /// Get path of nodes traversed in most recent forward pass
    pub fn path(&self) -> Vec<usize> {
        self.path.clone() 
    }

    /// Get all node references in graph
    pub fn nodes(&self) -> &Vec<RefCell<Box<dyn Node<T>>>> {
        &self.nodes
    }

    /// Borrow mutable reference of a node on the graph with specific index
    pub fn node(&self, index: usize) -> &RefCell<Box<dyn Node<T>>> {
        &self.nodes[index]
    }

    pub fn add_node(&mut self, node: Box<dyn Node<T>>) -> usize {
        self.nodes.push(RefCell::new(node));
        self.adj_list.insert(self.current_node_idx, HashSet::new());
        self.current_node_idx += 1;
        self.current_node_idx - 1
    }

    /// Create binary node with two inputs for operation
    pub fn binary(
        &mut self,
        lhs: Box<dyn Node<T>>,
        rhs: Box<dyn Node<T>>,
        op: Box<dyn Node<T>>) -> &mut Dendrite<T> {

        let lhs_idx = self.add_node(lhs); 
        let rhs_idx = self.add_node(rhs); 
        let op_idx = self.add_node(op); 

        if let Some(set) = self.adj_list.get_mut(&op_idx) {
            set.insert(lhs_idx);
            set.insert(rhs_idx); 
        } else {
            panic!("Error adding neighbors to add operation"); 
        }

        if let Some(set) = self.adj_list.get_mut(&lhs_idx) {
            set.insert(op_idx); 
        }

        if let Some(set) = self.adj_list.get_mut(&rhs_idx) {
            set.insert(op_idx); 
        } 

        self
    }

    /// Create unary node with one inputs for operation
    pub fn unary(
        &mut self,
        rhs: Box<dyn Node<T>>,
        op: Box<dyn Node<T>>) -> &mut Dendrite<T> {

        /*
        let lhs_idx = self.add_node(lhs); 
        let rhs_idx = self.add_node(rhs); 
        let op_idx = self.add_node(op); 

        if let Some(set) = self.adj_list.get_mut(&op_idx) {
            set.insert(lhs_idx);
            set.insert(rhs_idx); 
        } else {
            panic!("Error adding neighbors to add operation"); 
        }

        if let Some(set) = self.adj_list.get_mut(&lhs_idx) {
            set.insert(op_idx); 
        }

        if let Some(set) = self.adj_list.get_mut(&rhs_idx) {
            set.insert(op_idx); 
        } */

        self
    }


    /*
    /// Get node index of most recently computed node
    pub fn prev_node_idx(&self) -> usize {
        self.prev_node_idx 
    }

    /// Index of the next node ready for computation
    pub fn next_node_idx(&self) -> usize {
        self.next_node_idx 
    }


    /// Borrow mutable reference of a node on the graph with specific index
    pub fn node(&self, index: usize) -> &RefCell<Node<T>> {
        &self.nodes[index]
    }

    /// Borrow mutable reference of last node in the graph
    pub fn curr_node(&self) -> &RefCell<Node<T>> {
        &self.nodes[self.nodes.len() - 1]
    }

    /// Get all node references in graph
    pub fn nodes(&self) -> &Vec<RefCell<Node<T>>> {
        &self.nodes
    }

    /// Get successors (neighbors) of a specific node index
    pub fn successors(&self, index: usize) -> HashSet<usize> {
        self.adj_list.get(&index).unwrap().clone()
    }

    /// Create node on graph with 1 input (binary operation)
    pub fn unary(
        &mut self, 
        rhs: T, 
        op: Box<dyn Operation<T>>) -> Result<&mut Dendrite<T>, GraphError> {

        let node = Node::unary(rhs, op);

        self.adj_list.insert(
            self.current_node_idx(),
            HashSet::new()
        );
        self.nodes.push(RefCell::new(node));

        if self.prev_node_idx() == self.current_node_idx() {
            return Err(GraphError::UnaryOperation); 
        }
        
        if let Some(set) = self.adj_list.get_mut(&self.prev_node_idx) {
            set.insert(self.current_node_idx); 
        } else {
            // we shouldn't get to this error if the first one is thrown
            return Err(GraphError::NodeRelation);
        }

        self.prev_node_idx = self.current_node_idx();
        self.current_node_idx += 1;
        Ok(self)
    }

    /// recursive solution for forward pass
    pub fn forward_nodes(
        &mut self, 
        node_idx: usize, 
        prev_node_idx: Option<usize>) {

        let node = self.node(node_idx);
        if prev_node_idx.is_some() {
            let prev_node = self.node(prev_node_idx.unwrap());
            node.borrow_mut().forward(Some(&prev_node.borrow_mut()));
        } else {
            node.borrow_mut().forward(None); 
        }

        self.path.push(node_idx); 

        if node_idx == self.nodes.len() - 1 {
            return;
        }

        for item in self.successors(node_idx) {
            self.forward_nodes(item, Some(node_idx));
        }
    }

    /// Pass loss backward to graph to update gradients
    pub fn backward_nodes(
        &mut self,
        node_idx: usize,
        prev_node_idx: Option<usize>,
        upstream: Tensor<T>) {

        // fetch the node before the current node
        
        // stop recursion if node before current node is 0
        


    }

    /// Forward all nodes in the graph
    pub fn forward(&mut self) {
        self.forward_nodes(0, None); 
    } */


}

