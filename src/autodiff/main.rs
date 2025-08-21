use std::io::Write; 

use chrono::Local;
use ndarray::{arr2, Array2}; 

use dendritic_autodiff::graph::*;

use crate::autodiff::operations::activation::*; 
use crate::autodiff::operations::arithmetic::*; 
use crate::autodiff::operations::loss::*;

use env_logger; 

#[allow(dead_code)]
fn working_lin_regression_example() {

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
    graph.mul(vec![x, w]); 
    graph.add(vec![b]);
    graph.mse(y.clone());

    // mark parameter (temporary bad way for now)
    graph.add_parameter(1);
    graph.add_parameter(3);

    for _epoch in 0..1000 {

        graph.forward();

        let loss_node = graph.node(6);
        let loss = loss_node.output();
        println!("Loss: {:?}", loss.as_slice().unwrap()); 
        
        graph.backward();
    
        for var_idx in graph.parameters() {
            let var = graph.node(var_idx);
            let grad = var.grad() * (lr / y.len() as f64);
            let delta = var.output() - grad;
            graph.mut_node_output(var_idx, delta.clone());
        }


    }

    println!("{:?}", graph.node(4)); 


}

#[allow(dead_code)]
fn mlp_integration() {

    let lr: f64 = 0.01;
    let w1 = Array2::<f64>::zeros((2, 3));
    let b1 = Array2::<f64>::zeros((1, 3));
    let w2 = Array2::<f64>::zeros((3, 1));
    let b2 = Array2::<f64>::zeros((1, 1));

    let x = arr2(&[
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]);

    let y = arr2(&[[0.0],[0.0],[1.0],[1.0]]);

    let mut graph = ComputationGraph::new();

    //layer 1
    graph.mul(vec![x, w1]); 
    graph.add(vec![b1]);
    graph.sigmoid();

    // layer 2
    graph.mul(vec![w2]); 
    graph.add(vec![b2]);
    graph.sigmoid(); 

    // loss
    graph.mse(y.clone());

    // indicate which nodes are parameters
    graph.add_parameter(1); 
    graph.add_parameter(3); 
    graph.add_parameter(6); 
    graph.add_parameter(8); 

    for _epoch in 0..1 {
        
        graph.forward();

        let loss_node = graph.curr_node();
        let loss = loss_node.output();
        println!("Loss: {:?}", loss.as_slice().unwrap()); 

        graph.backward();

        for var_idx in graph.parameters() {
            let var = graph.node(var_idx);
            let grad = var.grad() * (lr as f64);
            let delta = var.output() - grad;
            graph.mut_node_output(var_idx, delta.clone());
        }

    }

}



fn main() {

    env_logger::builder()
    .format(|buf, record| {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        writeln!(buf, "{}:{} {}", log_time, record.level(), record.args())
    }).init();

    let b1 = Array2::<f64>::ones((1, 1));
    let w1 = Array2::<f64>::zeros((2, 1));

    // multi class
    let x1 = arr2(&[
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 1.0],   // Class 0
        [4.0, 4.5],
        [4.5, 4.8],
        [5.0, 5.2],   // Class 1
        [7.0, 7.5],
        [7.5, 8.0],
        [8.0, 8.5],   // Class 2
    ]);

    let y1 = arr2(&[
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [1.0],
        [2.0],
        [2.0],
        [2.0]
    ]);
    

    let mut graph = ComputationGraph::new();

    graph.mul(vec![x1, w1]); 
    graph.add(vec![b1]);
    graph.softmax();
    graph.cce(y1); 

    graph.forward(); 


}
