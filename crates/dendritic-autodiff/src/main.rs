use ndarray::prelude::*; 
use ndarray::{arr2, Array2}; 
use dendritic_autodiff::node::{Node};
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::{
    ComputationGraph,
    UnaryOperation, 
    BinaryOperation
}; 


pub fn mse(y_true: Array2<f64>, y_pred: Array2<f64>) -> f64 {

    if y_true.len() != y_pred.len() {
        panic!("Values for mse do not match in size"); 
    }

    let diff = y_true.clone() - y_pred; 
    let squared = diff.mapv(|x| x * x); 
    let sum = squared.sum(); 
    sum * (1.0/y_true.len() as f64)
}


fn main() {

    let lr: f64 = 0.01; 
    let w = Array2::<f64>::zeros((3, 1));
    let b = Array2::<f64>::zeros((1, 1));

    let x = arr2(&[
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ]);

    let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]); 
    
    let mut graph = ComputationGraph::new();
    graph.mul(x, w); 
    graph.u_add(b);
    graph.mse(y.clone());

    for epoch in 0..1000 {

        graph.forward();

        let loss = graph.node(6).output(); 
        println!("LOSS: {:?}", loss); 

        graph.backward();

        let x = graph.node(0); 
        let mut w = graph.node(1); 
        let mut b = graph.node(3);

        let w_grad = w.grad() * (lr / y.len() as f64); 
        let dw = w.output() - w_grad; 
        w.set_output(dw);

        //let b_grad = b.grad().sum_axis(Axis(0)); 
        let db = b.grad() * (lr / y.len() as f64);
        b.set_output(db);

    }


}
