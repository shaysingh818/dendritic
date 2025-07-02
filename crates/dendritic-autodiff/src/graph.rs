use std::fmt::Debug;
use std::fs::File;
use std::fs; 
use std::path::Path;
use std::io::{BufWriter, Write}; 
use std::hash::{DefaultHasher, Hash, Hasher}; 
use std::collections::{HashMap, HashSet}; 
use std::cell::{RefCell}; 

use polars::prelude::*; 
use log::{debug, info, warn}; 
use ndarray::Array2;
use serde::{Serialize, Deserialize}; 

use crate::node::{Node, NodeSerialization};
use crate::error::{GraphError};
use crate::registry::*; 
use crate::operations::base::*;


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
    pub operations: Vec<usize>,

    /// Mapping of strings to behavior traits for operations
    pub registry: HashMap<String, Box<dyn Operation<T>>> 
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraphMetadata {

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

    /// Create new instance of computation graph structure metadata
    pub fn serialize(&self) -> ComputationGraphMetadata {

        ComputationGraphMetadata {
            path: self.path.clone(),
            curr_node_idx: self.curr_node_idx,
            variables: self.variables.clone(),
            operations: self.operations.clone(),
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

    /// Get node indexes of variables in the computation expression
    pub fn parameters(&self) -> Vec<usize> {
        let mut var_idxs: Vec<usize> = Vec::new(); 
        for (idx, item) in self.nodes.iter().enumerate() {
            if item.is_param {
                var_idxs.push(idx);
            }
        }
        var_idxs
    }   

    /// Get node indexes of operations in the computation expression
    pub fn operations(&self) -> Vec<usize> {
        self.operations.clone()
    }

    /// Mark specific node index as a parameter value
    pub fn add_parameter(&mut self, node_idx: usize) {
        self.nodes[node_idx].is_param = true;
    }

    /// Mark specific node index as a parameter value
    pub fn register(&mut self, key: &str, op: Box<dyn Operation<T>>) {
        self.registry.insert(key.to_string(), op); 
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

        info!("Starting forward pass..."); 
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

        info!("Starting backward pass..."); 
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
 
        } else {

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

    /// Create node that applies functions to previous node
    pub fn function(
        &mut self, 
        op: Box<dyn Operation<T>>) -> &mut ComputationGraph<T> {
 
        let curr_node = self.curr_node_idx as usize;
        let prev_node = self.nodes[self.curr_node_idx as usize].clone(); 
        self.add_node(
            Node::unary(curr_node, op)
        );
        self.operations.push(self.curr_node_idx as usize); 

        let new_node_idx = self.curr_node_idx as usize;
        self.nodes[new_node_idx].set_output(prev_node.output());

        self.add_upstream_node(
            curr_node, 
            vec![self.curr_node_idx as usize]
        );
        self
    }

}


pub trait GraphConstruction<T> {
    
    fn new() -> Self;

}


macro_rules! graph_constructor {

    ($t:ty) => {

        impl GraphConstruction<$t> for ComputationGraph<$t> {

            fn new() -> Self {

                let mut graph = ComputationGraph {
                    nodes: vec![],
                    path: vec![],
                    curr_node_idx: -1,
                    variables: vec![],
                    operations: vec![],
                    registry: HashMap::new()
                };
                graph.register_default_operations(); 
                graph
            }

        }

    }

}

graph_constructor!(f64);
graph_constructor!(Array2<f64>); 


macro_rules! graph_serialize {


    ($t:ty) => {

        impl ComputationGraph<$t> {

            pub fn save(&self, namespace: &str) -> std::io::Result<()> {

                let mut hasher = DefaultHasher::new();
                hasher.write(namespace.as_bytes()); 
                let hashed = hasher.finish(); 

                let nodes = self.nodes.clone(); 
                let node_namespace = format!("{namespace}/{hashed}_nodes.json"); 
                let metadata = format!("{namespace}/{hashed}_metadata.json");

                if !Path::new(namespace).exists() {
                    match fs::create_dir(namespace) {
                        Ok(_) => debug!("namespace partition created"),
                        Err(e) => eprintln!(
                            "Failed to create namespace: {}", e
                        ),
                    }
                } else {
                    debug!(
                        "Namepspace partition already exists {}", 
                        namespace
                    );
                }

                let node_file = match File::create(node_namespace) {
                    Ok(file) => file,
                    Err(err) => {
                        return Err(err);
                    }
                };

                let metadata_file = match File::create(metadata) {
                    Ok(file) => file,
                    Err(err) => {
                        return Err(err);
                    }
                };  

                let mut node_writer = BufWriter::new(node_file);
                let mut node_vec = Vec::new();
                for node in nodes {
                    node_vec.push(node.serialize());
                }
                serde_json::to_writer_pretty(&mut node_writer, &node_vec);
                node_writer.flush()?; 

                let mut metadata_writer = BufWriter::new(metadata_file);
                let metadata_obj = self.serialize();
                serde_json::to_writer_pretty(
                    &mut metadata_writer, 
                    &metadata_obj
                );
                metadata_writer.flush()?; 

                Ok(())
            
            }

            pub fn load(&self, filepath: &str) -> std::io::Result<()> {
                Ok(())
            }

        }


    }

}

graph_serialize!(f64); 
graph_serialize!(Array2<f64>);

