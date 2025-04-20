use std::collections::{HashMap, HashSet}; 
use std::cell::{RefCell}; 
use crate::node::{Node};
use crate::error::{GraphError};

/// A dendrite is an instance of expression stored in a computation graph.
/// A dendrite stores an adjacency list of nodes (operations) in the
/// computation graph. Since nodes are not clonable and only references
/// of nodes can be used, the array of nodes is stored as smart pointers
/// that allow for interior mutability. 
#[derive(Debug)]
pub struct Dendrite<T> {

    /// references to node operations in the graph
    pub nodes: Vec<Node<T>>,

    /// Temporary list to store path traversals in graph
    pub path: Vec<usize>,
}

impl<T: Clone+ Default> Dendrite<T> {

    /// Create new instance of computation graph structure
    pub fn new() -> Self {

        Dendrite {
            nodes: vec![],
            path: vec![],
        }
    }


    /// Get path of nodes traversed in most recent forward pass
    pub fn path(&self) -> Vec<usize> {
        self.path.clone() 
    }

    pub fn node(&self, idx: usize) -> Node<T> {
        self.nodes[idx].clone()
    }

    pub fn add_node(&mut self, node: Node<T>) {
        self.nodes.push(node);
    }

    pub fn forward_node(&mut self, idx: usize) {
        let inputs = self.nodes[idx].inputs.clone(); 
        let output = self.nodes[idx].forward(&self.nodes, inputs);
        self.nodes[idx].set_output(output); 
    }

    /*
    /// Borrow mutable reference of a node on the graph with specific index
    pub fn node(&self, index: usize) -> &dyn NodeTrait<T> {
        &*self.nodes[index]
    }

    /// Get successors (neighbors) of a specific node index
    pub fn successors(&self, index: usize) -> HashSet<usize> {
        self.adj_list.get(&index).unwrap().clone()
    }

    /// Create binary node with two inputs for operation
    pub fn binary(
        &mut self,
        lhs: T,
        rhs: T,
        op: Box<dyn Operation<T>>) -> &mut Dendrite<T> {

        let node = Node::binary(lhs, rhs, op);

        self.adj_list.insert(
            self.current_node_idx(),
            HashSet::new()
        );

        self.nodes.push(RefCell::new(node));
        self.prev_node_idx = self.current_node_idx();
        self.current_node_idx += 1;
        self
    }

    /// Create unary node with one inputs for operation
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

    /// recursive solution for forward pass
    pub fn backward_nodes(&mut self, node_idx: usize) {

        if node_idx == 0 {
            return; 
        }

        let curr_node = self.node(node_idx);

        for item in self.successors(node_idx) {
            let neighbor = self.node(item);
            curr_node.borrow_mut().backward(&mut neighbor.borrow_mut()); 
        }

        for item in self.successors(node_idx) {
            self.backward_nodes(item);
        }

    }

    /// Forward all nodes in the graph
    pub fn forward(&mut self) {
        self.forward_nodes(0, None); 
    }

    /// Backward pass for all nodes in the graph 
    pub fn backward(&mut self) {
        let node_idx: usize = self.current_node_idx - 1;
        self.backward_nodes(node_idx); 
    }

    /*



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
    */

}

