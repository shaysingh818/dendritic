use std::fmt::Debug;
use std::fs::File;
use std::fs; 
use std::io::{BufWriter, BufReader}; 
use std::hash::{DefaultHasher, Hasher}; 
use std::collections::HashMap; 

use log::info; 
use ndarray::Array2;
use serde::{Serialize, Deserialize}; 

use crate::autodiff::node::{Node, NodeSerialization, NodeSerialize};
use crate::autodiff::registry::*; 
use crate::autodiff::operations::base::*;


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


/// Serializable struct for computation graph that stores metadata
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

    /// Set output of specific node value and reference by index
    pub fn mut_node_operation(&mut self, idx: usize, op: Box<dyn Operation<T>>) {
        self.nodes[idx].set_operation(op);
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

        let node = &mut self.nodes[node_idx];
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
        for node in &mut nodes {
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


pub trait GraphSerialize<T> {
    
    fn save(&self, namespace: &str) -> std::io::Result<()>;

    fn load(filepath: &str) -> std::io::Result<ComputationGraph<T>>;
}


macro_rules! graph_serialize {


    ($t:ty) => {

        impl GraphSerialize<$t> for ComputationGraph<$t> {

            fn save(&self, namespace: &str) -> std::io::Result<()> {

                let mut hasher = DefaultHasher::new();
                hasher.write(namespace.as_bytes()); 
                let hashed = hasher.finish(); 

                let node_path = format!("{namespace}/{hashed}_nodes.json"); 
                let metadata_path = format!("{namespace}/{hashed}_metadata.json");

                fs::create_dir_all(namespace)?;

                {
                    let file = File::create(&node_path)?;
                    let writer = BufWriter::new(file);
                    let serialized_nodes: Vec<_> = self
                        .nodes
                        .iter()
                        .map(|n| n.serialize())
                        .collect();

                    serde_json::to_writer_pretty(writer, &serialized_nodes)?;
                }

                {
                    let file = File::create(&metadata_path)?;
                    let writer = BufWriter::new(file);
                    let metadata_obj = self.serialize();
                    serde_json::to_writer_pretty(writer, &metadata_obj)?;
                }

                Ok(())
            }

            fn load(filepath: &str) -> std::io::Result<ComputationGraph<$t>> {

                let mut node_file: Option<String> = None; 
                let mut metadata_file: Option<String> = None;

                for entry in fs::read_dir(filepath)? {
                    let path = entry?.path();

                    if path.is_file() {
                        if let Some(name) = path.to_str() {
                            if name.contains("_nodes") {
                                node_file = Some(name.to_string());
                            } else if name.contains("_metadata") {
                                metadata_file = Some(name.to_string());
                            }
                        }
                    }
                }
                
                let node_file = node_file.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound, 
                        "File containing serialized nodes not found"
                    )
                })?;

                let metadata_file = metadata_file.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound, 
                        "File containing graph metadata not found"
                    )
                })?;

                let nodes: Vec<NodeSerialize<$t>> = {
                    let file = File::open(&node_file)?;
                    let reader = BufReader::new(file);
                    serde_json::from_reader(reader)?
                };
                
                let g_metadata: ComputationGraphMetadata = {
                    let file = File::open(&metadata_file)?;
                    let reader = BufReader::new(file);
                    serde_json::from_reader(reader)?
                };

                let mut graph = ComputationGraph {
                    nodes: vec![],
                    path: g_metadata.path,
                    curr_node_idx: g_metadata.curr_node_idx,
                    variables: g_metadata.variables,
                    operations: g_metadata.operations,
                    registry: HashMap::new()
                };
                graph.register_default_operations();


                for node in nodes.iter() {
                    let item = Node::load(
                        node.clone(), 
                        graph.registry.clone()
                    )?;
                    graph.nodes.push(item); 
                }

                Ok(graph)
            }
        }
    }

}

graph_serialize!(f64); 
graph_serialize!(Array2<f64>);

