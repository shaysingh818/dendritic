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


/// Trait for constructing computation graphs
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

/// Trait for serializing computation graphs
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


#[cfg(test)]
mod graph_test {

    use std::fs;
    use std::fs::File; 
    use crate::autodiff::node::*; 
    use crate::autodiff::operations::activation::*; 
    use crate::autodiff::operations::arithmetic::*; 
    use crate::autodiff::operations::loss::*; 
    use crate::autodiff::graph::*; 
    use ndarray::{arr2};


    #[test]
    fn test_graph_instantiation() {

        let graph: ComputationGraph<f64> = ComputationGraph::new();

        assert_eq!(graph.nodes().len(), 0); 
        assert_eq!(graph.curr_node_idx(), -1);
        assert_eq!(graph.path().len(), 0);
        assert_eq!(graph.variables().len(), 0); 
        assert_eq!(graph.operations().len(), 0); 
    }

    #[test]
    fn test_graph_binary_node() {

        let a = Some(5.0); 
        let b = Some(10.0); 

        let mut graph = ComputationGraph::new(); 
        graph.binary(a, b, Box::new(Add));

        assert_eq!(graph.nodes().len(), 3); 
        assert_eq!(graph.curr_node_idx(), 2);

        let a_val = graph.node(0); 
        let b_val = graph.node(1); 
        let add_node = graph.node(2);

        assert_eq!(a_val.upstream(), vec![2]); 
        assert_eq!(b_val.upstream(), vec![2]); 
        assert_eq!(a_val.inputs().len(), 0); 
        assert_eq!(b_val.inputs().len(), 0);

        assert_eq!(a_val.output(), 5.0); 
        assert_eq!(b_val.output(), 10.0);

        assert_eq!(add_node.upstream().len(), 0); 
        assert_eq!(add_node.inputs().len(), 2); 
        assert_eq!(add_node.inputs(), vec![0, 1]);
        assert_eq!(add_node.output(), 0.0); 
    }

    #[test]
    fn test_graph_unary_node() -> Result<(), Box<dyn std::error::Error>>  {

        let a = Some(5.0); 
        let b = Some(10.0);
        let c = 100.0; 

        let mut graph = ComputationGraph::new(); 
        graph.binary(a, b, Box::new(Add));
        graph.unary(c, Box::new(Add)); 

        assert_eq!(graph.nodes().len(), 5); 
        assert_eq!(graph.curr_node_idx(), 4); 

        let a_val = graph.node(0); 
        let b_val = graph.node(1); 
        let add = graph.node(2); 
        let c_val = graph.node(3); 
        let add_2 = graph.node(4);

        assert_eq!(a_val.upstream(), vec![2]); 
        assert_eq!(b_val.upstream(), vec![2]); 
        assert_eq!(a_val.inputs().len(), 0); 
        assert_eq!(b_val.inputs().len(), 0);
        assert_eq!(a_val.output(), 5.0); 
        assert_eq!(b_val.output(), 10.0);

        assert_eq!(add.upstream().len(), 1);
        assert_eq!(add.upstream(), vec![4]); 
        assert_eq!(add.inputs().len(), 2); 
        assert_eq!(add.inputs(), vec![0, 1]);
        assert_eq!(add.output(), 0.0);

        assert_eq!(c_val.inputs().len(), 0); 
        assert_eq!(c_val.upstream().len(), 1); 
        assert_eq!(c_val.upstream(), vec![4]); 
        assert_eq!(c_val.output(), 100.0);

        assert_eq!(add_2.inputs().len(), 2); 
        assert_eq!(add_2.inputs(), vec![2, 3]); 
        assert_eq!(add_2.upstream().len(), 0); 
        assert_eq!(add_2.output(), 0.0); 

        Ok(())
    }


    #[test]
    fn test_graph_fn_node() -> Result<(), Box<dyn std::error::Error>>  {

        let a = arr2(&[[0.0],[0.0],[0.0],[0.0]]);
        let b = arr2(&[[0.0],[0.0],[0.0],[0.0]]);
        let y = arr2(&[[1.0],[1.0],[1.0],[1.0]]); 

        let mut graph = ComputationGraph::new();
        graph.add(vec![a, b]);
        graph.function(Box::new(Sigmoid));
        graph.mse(y.clone()); 

        assert_eq!(graph.nodes().len(), 6); 
        assert_eq!(graph.variables(), vec![0,1,4]); 
        assert_eq!(graph.operations(), vec![2,3,5]);

        graph.forward();

        assert_eq!(
            graph.node(2).output(), 
            arr2(&[[0.0],[0.0],[0.0],[0.0]])
        );

        assert_eq!(
            graph.node(3).output(),
            arr2(&[[0.5],[0.5],[0.5],[0.5]])
        );

        graph.backward(); 

        assert_eq!(
            graph.node(3).grad(),
            arr2(&[[-0.125],[-0.125],[-0.125],[-0.125]])
        );

        assert_eq!(
            graph.node(2).grad(),
            arr2(&[[-0.125],[-0.125],[-0.125],[-0.125]])
        );

        assert_eq!(
            graph.node(1).grad(),
            arr2(&[[-0.125],[-0.125],[-0.125],[-0.125]])
        );

        assert_eq!(
            graph.node(0).grad(),
            arr2(&[[-0.125],[-0.125],[-0.125],[-0.125]])
        );

        Ok(())
    }

    
    #[test]
    fn test_graph_operation_relationships() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]); 

        assert_eq!(graph.nodes().len(), 9);
        assert_eq!(graph.path().len(), 0);
        assert_eq!(graph.curr_node_idx(), 8);

        let val1 = graph.node(0);
        assert_eq!(val1.upstream(), vec![2]); 
        assert_eq!(val1.inputs().len(), 0); 

        let val2 = graph.node(1);
        assert_eq!(val2.upstream(), vec![2]); 
        assert_eq!(val2.inputs().len(), 0); 

        let add_node = graph.node(2);
        assert_eq!(add_node.upstream(), vec![4]); 
        assert_eq!(add_node.inputs().len(), 2); 
        assert_eq!(add_node.inputs(), vec![0, 1]); 

        let val3 = graph.node(3);
        assert_eq!(val3.upstream(), vec![4]); 
        assert_eq!(val3.inputs().len(), 0); 

        let u_add_node = graph.node(4);
        assert_eq!(u_add_node.upstream(), vec![6]); 
        assert_eq!(u_add_node.inputs().len(), 2);
        assert_eq!(u_add_node.inputs(), vec![2, 3]);

        let val4 = graph.node(5);
        assert_eq!(val4.upstream(), vec![6]); 
        assert_eq!(val4.inputs().len(), 0);

        let u_mul_node = graph.node(6);
        assert_eq!(u_mul_node.upstream(), vec![8]); 
        assert_eq!(u_mul_node.inputs().len(), 2);
        assert_eq!(u_mul_node.inputs(), vec![4, 5]);
        
        let val5 = graph.node(7);
        assert_eq!(val5.upstream(), vec![8]); 
        assert_eq!(val5.inputs().len(), 0);

        let u_sub_node = graph.node(8);
        assert_eq!(u_sub_node.upstream().len(), 0); 
        assert_eq!(u_sub_node.inputs().len(), 2);
        assert_eq!(u_sub_node.inputs(), vec![6, 7]);

    }

    #[test]
    fn test_graph_forward_evaluate_scalar() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]); 

        graph.forward(); 

        assert_eq!(graph.path().len(), 4);
        assert_eq!(
            graph.path(),
            vec![2, 4, 6, 8]
        );

        let expected_outputs = vec![15.0, 115.0, 2300.0, 2290.0];

        for (idx, node) in graph.path().iter().enumerate() {
            let node_output = graph.node(*node);
            assert_eq!(node_output.output(), expected_outputs[idx]); 
        }
    }

    #[test]
    fn test_graph_backward_evaluate_scalar() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]); 

        graph.forward(); 

        graph.backward();

        assert_eq!(graph.path().len(), 4);
        assert_eq!(
            graph.path(),
            vec![2, 4, 6, 8]
        );

        let mut path = graph.path().clone(); 
        path.reverse();

        let vars = graph.variables(); 
        let ops = graph.operations();

        let expected_var_grads = vec![1.0, 1.0, 1.0, 115.0, 1.0];
        let expected_op_grads = vec![1.0, 20.0, 1.0, 0.0];

        for (idx, var) in vars.iter().enumerate() {
            let node = graph.node(*var); 
            assert_eq!(node.grad(), expected_var_grads[idx]); 
        }

        for (idx, op) in ops.iter().enumerate() {
            let node = graph.node(*op); 
            assert_eq!(node.grad(), expected_op_grads[idx]); 
        }

    }

    #[test]
    fn test_graph_op_registry() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]);

        let registry_keys = graph.registry.keys(); 
        let mut keys_vec: Vec<String> = registry_keys.cloned().collect();

        let mut expected = vec![
            "Mul", "Sub", "Add", "DefaultValue", 
            "Tanh", "BinaryCrossEntropy", "CategoricalCrossEntropy", 
            "DefaultLossFunction",
            "MSE", "Sigmoid"
        ];

        keys_vec.sort(); 
        expected.sort(); 

        assert_eq!(keys_vec, expected); 
    }


    #[test]
    fn test_graph_save() -> std::io::Result<()> {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]);

        let _ = graph.save("testing");

        let mut node_file: Option<String> = None; 
        let mut metadata_file: Option<String> = None; 

        for entry in fs::read_dir("testing")? {

            let entry = entry?;
            let path = entry.path();

            if path.is_file() {

                if let Some(file_name) = path.to_str() {
 
                    if file_name.contains("_nodes") {
                        node_file = Some(file_name.to_string()); 
                    }

                    if file_name.contains("_metadata") {
                        metadata_file = Some(file_name.to_string()); 
                    }

                }

            }

        }

        let node_file_read = File::open(node_file.unwrap())?; 
        let metadata_file_read = File::open(metadata_file.unwrap())?; 

        let nodes: Vec<NodeSerialize<f64>> = serde_json::from_reader(
            node_file_read
        )?;

        let mut nodes_vec: Vec<Node<f64>> = Vec::new(); 
        for node in nodes.iter() {
            let item = Node::load(node.clone(), graph.registry.clone())?;
            nodes_vec.push(item); 
        }

        assert_eq!(nodes_vec.len(), graph.nodes().len());

        let g_metadata: ComputationGraphMetadata = serde_json::from_reader(
            metadata_file_read
        )?;

        assert_eq!(g_metadata.path, graph.path());
        assert_eq!(g_metadata.curr_node_idx, graph.curr_node_idx()); 
        assert_eq!(g_metadata.variables, graph.variables());
        assert_eq!(g_metadata.operations, graph.operations());

        fs::remove_dir_all("testing")?; 
        Ok(())
    }


    #[test]
    fn test_graph_load() -> std::io::Result<()> {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]);

        let _ = graph.save("sample_saved_graph");

        let loaded_graph: ComputationGraph<f64> = ComputationGraph::load("sample_saved_graph").unwrap();

        assert_eq!(loaded_graph.nodes().len(), graph.nodes().len()); 
        assert_eq!(loaded_graph.path(), graph.path());
        assert_eq!(loaded_graph.variables(), graph.variables()); 
        assert_eq!(loaded_graph.curr_node_idx(), graph.curr_node_idx());
        assert_eq!(loaded_graph.operations(), graph.operations());

        fs::remove_dir_all("sample_saved_graph")?; 
        Ok(())

    }



}

