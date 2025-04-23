use std::fmt::Debug; 
use std::collections::{HashMap, HashSet}; 
use std::cell::{RefCell}; 
use crate::node::{Node};
use crate::ops::Operation; 
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

    /// Current node index on computation
    pub curr_node_idx: i64,
}

impl<T: Clone + Default + Debug> Dendrite<T> {

    /// Create new instance of computation graph structure
    pub fn new() -> Self {

        Dendrite {
            nodes: vec![],
            path: vec![],
            curr_node_idx: -1,
        }
    }


    /// Get path of nodes traversed in most recent forward pass
    pub fn path(&self) -> Vec<usize> {
        self.path.clone() 
    }

    pub fn node(&self, idx: usize) -> Node<T> {
        self.nodes[idx].clone()
    }

    pub fn curr_node(&self) -> Node<T> {
        self.nodes[self.curr_node_idx as usize].clone()
    }

    pub fn binary(
        &mut self, 
        lhs: Option<T>, 
        rhs: Option<T>, 
        op: Operation<T>) -> &mut Dendrite<T> {

        match lhs {
            Some(input) => self.add_node(Node::val(input)),
            None => {

            }
        }

        match rhs {
            Some(input) => self.add_node(Node::val(input)),
            None => {

            }
        }
            

        //self.add_node(Node::val(lhs)); 
        //self.add_node(Node::val(rhs)); 
        self.add_node(
            Node::binary(
                (self.curr_node_idx - 1) as usize, 
                self.curr_node_idx as usize, 
                op
            )
        );

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

    pub fn unary(&mut self, rhs: T, op: Operation<T>) -> &mut Dendrite<T> {

        self.add_node(Node::val(rhs)); 
        self.add_node(
            Node::binary(
                (self.curr_node_idx - 1) as usize, 
                self.curr_node_idx as usize, 
                op
            )
        );

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

    pub fn add_node(&mut self, node: Node<T>) {
        self.nodes.push(node);
        self.curr_node_idx += 1;
    }

    pub fn add_upstream_node(
        &mut self,
        node_idx: usize,
        upstream_vals: Vec<usize>) {

        let mut node = &mut self.nodes[node_idx];
        for upstream in upstream_vals {
            node.add_upstream(upstream); 
        }
    }

    pub fn forward_node(&mut self, idx: usize) {
        let inputs = self.nodes[idx].inputs(); 
        let output = self.nodes[idx].forward(&self.nodes, inputs, idx);
        self.nodes[idx].set_output(output); 
    }

    pub fn backward_node(&mut self, idx: usize) {
        let inputs = self.nodes[idx].inputs.clone();
        let node_call = self.nodes[idx].clone(); 
        node_call.backward(&mut self.nodes, inputs, idx);
    }

    pub fn backward(&mut self, node_idx: usize) {

        if node_idx == 0 {
            return; 
        }

        self.backward_node(node_idx); 

        let inputs = self.nodes[node_idx].inputs();
        for item in inputs {
            self.backward(item);
        }
    }


    pub fn forward(&mut self, node_idx: usize) {
 
        if node_idx > self.nodes.len() - 1 {
            return; 
        }

        self.forward_node(node_idx); 

        let upstream = self.nodes[node_idx].upstream();  
        for item in upstream {
            self.forward(item);
        }

    }

}

