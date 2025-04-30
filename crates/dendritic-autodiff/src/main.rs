use dendritic_autodiff::ops::{Operation};
use dendritic_autodiff::node::{Node};
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::{Dendrite}; 

fn main() {

    // forward expression (with shared inputs)
    //
    let a = Some(3.0);
    let b = Some(1.0);
    let c = Some(-2.0); 
    
    /*
    let mut graph: Dendrite<f64> = Dendrite::new();
    graph.binary(Some(2.0), b, Operation::mul());
    graph.unary(a.unwrap(), Operation::add()); 
    graph.unary(c.unwrap(), Operation::mul()); 
    */
    //graph.forward(0); 

    //graph.backward(6); 

    let mut graph2: Dendrite<f64> = Dendrite::new(); 
    graph2.binary(a, b, Operation::add()); 
    graph2.binary(b, Some(1.0), Operation::add());
    graph2.binary(None, None, Operation::mul());

    graph2.forward();
    println!("{:?}", graph2.node(2)); 
    println!("{:?}", graph2.node(5)); 
    println!("{:?}", graph2.node(6)); 
}
