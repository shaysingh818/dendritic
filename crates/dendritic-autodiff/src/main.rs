use std::cell::{RefCell}; 
use std::borrow::Borrow; 
use dendritic_autodiff::ops::{Add, Operation};
use dendritic_autodiff::node::{Node};
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::{Dendrite}; 
use dendritic_autodiff::binary::*;
use dendritic_autodiff::unary::*;

fn main() {

    // forward expression (with shared inputs)
    //
    
    let mut graph: Dendrite<f64> = Dendrite::new(); 
    graph.add_node(Node::val(5.0)); 
    graph.add_node(Node::val(10.0));
    graph.add_node(Node::binary(0, 1, Operation::add()));
    graph.add_node(Node::val(20.0));
    graph.add_node(Node::binary(2, 3, Operation::add()));


    graph.forward_node(2);
    graph.forward_node(4);

    println!("{:?}", graph.node(2).output()); 
    println!("{:?}", graph.node(4).output()); 
    /*
    let val = nodes[2].forward(&nodes, nodes[2].inputs.clone()); 
    println!("{:?}", val);
    nodes[2].set_output(val); 

    let val_2 = nodes[4].forward(&nodes, nodes[4].inputs.clone()); 
    println!("{:?}", val_2); 
    */


    // Full backward pass with gradients updated
    
    // Expression structure stored in graph structure (vector of nodes)

}
