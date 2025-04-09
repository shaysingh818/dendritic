use dendritic_autodiff::node::{Node};
use dendritic_autodiff::ops::{Add};
use dendritic_autodiff::binary::*;
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::{Dendrite}; 
use std::cell::RefCell; 

fn main() {

    println!("Testing funcs");

    let mut torch = Dendrite::new();
    torch.add(10.0, 13.0);

    println!("Number of nodes: {:?}", torch.nodes().len());

    let a: Tensor<f64> = Tensor::new(&10.0); 
    let b: Tensor<f64> = Tensor::new(&11.0); 
    let op = Add::new(a.clone(), b.clone());

    let node_0 = torch.node(0); 
    let node_1 = torch.node(1); 

    node_0.borrow_mut().forward(); 
    node_1.borrow_mut().forward(); 

    println!("{:?}", torch.adj_list); 

    /*
    let node_0 = torch.node(0); 
    node_0.forward(); */ 







}
