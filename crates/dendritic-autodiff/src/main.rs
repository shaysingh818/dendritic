use dendritic_autodiff::node::{Node, Operation};
use dendritic_autodiff::ops::{Add, Sub, Mul, Div};
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::{Dendrite}; 
use dendritic_autodiff::binary::*;
use dendritic_autodiff::unary::*;

fn main() {

    let a = 3.0;
    let b = 1.0;
    let c = -2.0; 

    // Shared parameter example expression: (a+b) * (b+1)
    let mut torch = Dendrite::new();
    torch.add(a, b.clone());
    torch.add(b.clone(), 1.0); 

    // None shared parameter example
    let mut graph = Dendrite::new(); 
    graph.mul(2.0, b.clone()); 
    graph.u_add(a.clone());
    graph.u_mul(c); 

    graph.forward();

    println!("{:?}", graph.adj_list);

    let output = graph.curr_node().borrow_mut().output();
    println!("{:?}", output); 


}
