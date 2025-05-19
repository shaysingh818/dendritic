use std::fmt::Debug; 
use std::collections::{HashMap, HashSet}; 
use std::cell::{RefCell}; 
use crate::node::{Node};
use crate::error::{GraphError};
use ndarray::Array2;
use crate::ops::*; 

/// A dendrite is an instance of expression stored in a computation graph.
/// A dendrite stores an adjacency list of nodes (operations) in the
/// computation graph. Since nodes are not clonable and only references
/// of nodes can be used, the array of nodes is stored as smart pointers
/// that allow for interior mutability. 
pub struct ComputationGraph<T> {

    /// references to node operations in the graph
    pub nodes: Vec<Node<T>>,

    /// Temporary list to store path traversals in graph
    pub path: Vec<usize>,

    /// Current node index on computation
    pub curr_node_idx: i64,

    /// List of indices that point to raw variables in the graph
    pub variables: Vec<usize>,

    /// List of indices that point to operations in the graph
    pub operations: Vec<usize>
}

impl<T: Clone + Default + Debug> ComputationGraph<T> {

    /// Create new instance of computation graph structure
    pub fn new() -> Self {

        ComputationGraph {
            nodes: vec![],
            path: vec![],
            curr_node_idx: -1,
            variables: vec![],
            operations: vec![]
        }
    }


    /// Get path of nodes traversed in most recent forward pass
    pub fn path(&self) -> Vec<usize> {
        self.path.clone() 
    }

    /// Get node in graph at specific index of node array
    pub fn node(&self, idx: usize) -> Node<T> {
        self.nodes[idx].clone()
    }

    /// Return copy/clone of nodes array
    pub fn nodes(&self) -> Vec<Node<T>> {
        self.nodes.clone()
    }

    /// Set output of specific node value and reference by index
    pub fn mut_node_output(&mut self, idx: usize, val: T) {
        self.nodes[idx].set_output(val);
    }

    /// Get latest node created in the computation graph
    pub fn curr_node(&self) -> Node<T> {
        self.nodes[self.curr_node_idx as usize].clone()
    }

    /// Get index of latest node in the computation graph
    pub fn curr_node_idx(&self) -> i64 {
        self.curr_node_idx
    }

    /// Get node indexes of variables in the computation expression
    pub fn variables(&self) -> Vec<usize> { 
        self.variables.clone()
    }   

    /// Get node indexes of operations in the computation expression
    pub fn operations(&self) -> Vec<usize> {
        self.operations.clone()
    }

    /// Add node to current array of nodes
    pub fn add_node(&mut self, node: Node<T>) {
        self.nodes.push(node);
        self.curr_node_idx += 1;
    }

    /// Add upstream node index for current node value
    pub fn add_upstream_node(
        &mut self,
        node_idx: usize,
        upstream_vals: Vec<usize>) {

        let mut node = &mut self.nodes[node_idx];
        for upstream in upstream_vals {
            node.add_upstream(upstream); 
        }
    }

    /// Call forward operation on current node index
    pub fn forward_node(&mut self, idx: usize) {
        let output = self.nodes[idx].forward(&self.nodes, idx);
        self.nodes[idx].set_output(output); 
    }

    /// Call backward operation on current node index
    pub fn backward_node(&mut self, idx: usize) {
        let mut node_call = self.nodes[idx].clone(); 
        node_call.backward(&mut self.nodes, idx);
    }

    /// Perform forward pass on all nodes in the graph
    pub fn forward(&mut self) {
        let mut idx = 0;
        let mut nodes = self.nodes.clone(); 

        debug_log("Starting forward pass..."); 
        for mut node in &mut nodes {
            if node.inputs().len() > 0 {
                self.path.push(idx);
                self.forward_node(idx);
            }       
            idx += 1; 
        }
    }

    /// Perform backward pass on all nodes in the graph
    pub fn backward(&mut self) {

        if self.path.len() <= 0 {
            panic!("Forward pass path has not been completed yet"); 
        }
 
        let mut path_clone = self.path.clone(); 
        path_clone.reverse();

        debug_log("Starting backward pass..."); 
        for node_idx in path_clone {
            self.backward_node(node_idx); 
        }
    }

    /// Retrieve index of other binary index in unary operation
    pub fn binary_relation(&mut self) -> usize {
        for (idx, node) in &mut self.nodes.iter_mut().enumerate() {
            if node.inputs().len() != 0 {
                if idx != self.curr_node_idx as usize {
                    return idx;
                }
            }
        }
        0
    }

    /// Create a binary node relationship between 2 provided inputs
    pub fn binary(
        &mut self, 
        lhs: Option<T>, 
        rhs: Option<T>, 
        op: Box<dyn Operation<T>>) -> &mut ComputationGraph<T> {

        match lhs {
            Some(ref input) => {
                self.add_node(Node::val(input.clone()));
                self.variables.push(self.curr_node_idx as usize); 
            },
            None => {}
        }

        match rhs {
            Some(ref input) => {
                self.add_node(Node::val(input.clone()));
                self.variables.push(self.curr_node_idx as usize); 
            },
            None => {}
        }

        if lhs.is_none() && rhs.is_none() {

            let lhs_idx = self.binary_relation();
            let rhs_idx = self.curr_node_idx as usize;

            debug_log(
                &format!(
                    "Set input indexes: ({:?}, {:?})",
                    lhs_idx, rhs_idx
                )
            );

            self.add_node(Node::binary(lhs_idx, rhs_idx, op));
            self.operations.push(self.curr_node_idx as usize); 

            self.add_upstream_node(
                lhs_idx, 
                vec![self.curr_node_idx as usize]
            );
            self.add_upstream_node(
                rhs_idx, 
                vec![self.curr_node_idx as usize]
            );

            debug_log(
                &format!(
                    "Set upstream indexes: ({:?}, {:?}) -> {:?}",
                    lhs_idx, rhs_idx, self.curr_node_idx
                )
            );
            
        } else {

            debug_log(
                &format!(
                    "Set input indexes: ({:?}, {:?})",
                    self.curr_node_idx - 1, self.curr_node_idx
                )
            );

            self.add_node(
                Node::binary(
                    (self.curr_node_idx - 1) as usize, 
                    self.curr_node_idx as usize, 
                    op
                )
            );
            self.operations.push(self.curr_node_idx as usize); 

            self.add_upstream_node(
                (self.curr_node_idx - 2) as usize, 
                vec![self.curr_node_idx as usize]
            );

            self.add_upstream_node(
                (self.curr_node_idx - 1) as usize, 
                vec![self.curr_node_idx as usize]
            );

            debug_log(
                &format!(
                    "Set upstream indexes: ({:?}, {:?}) -> {:?}",
                    self.curr_node_idx-2, self.curr_node_idx-1, self.curr_node_idx
                )
            );
        }

        self
    }

    /// Create unary node relationship with only one input value provided
    pub fn unary(
        &mut self, 
        rhs: T, 
        op: Box<dyn Operation<T>>) -> &mut ComputationGraph<T> {

        self.add_node(Node::val(rhs));
        self.variables.push(self.curr_node_idx as usize); 

        self.add_node(
            Node::binary(
                (self.curr_node_idx - 1) as usize, 
                self.curr_node_idx as usize, 
                op
            )
        );
        self.operations.push(self.curr_node_idx as usize); 

        self.add_upstream_node(
            (self.curr_node_idx - 2) as usize, 
            vec![self.curr_node_idx as usize]
        );

        self.add_upstream_node(
            (self.curr_node_idx - 1) as usize, 
            vec![self.curr_node_idx as usize]
        );

        self
    }


}

