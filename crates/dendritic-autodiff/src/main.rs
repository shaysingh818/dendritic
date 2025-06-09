use ndarray::prelude::*; 
use ndarray::{arr2, Array2}; 
use dendritic_autodiff::node::{Node};
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::*;

use dendritic_autodiff::operations::activation::*; 
use dendritic_autodiff::operations::arithmetic::*; 
use dendritic_autodiff::operations::loss::*; 

use polars::prelude::*; 

fn write_df_to_file() -> Result<(), Box<dyn std::error::Error>>   {
    let s = Series::new("values".into(), &[1, 2, 3]);
    let df = DataFrame::new(vec![s.into()])?;
    //ParquetWriter::new("data.parquet").finish(&mut df.clone())?;
    Ok(())
}

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

    for epoch in 0..1000 {

        graph.forward();

        let loss_node = graph.node(6);
        let loss = loss_node.output();
        println!("Loss: {:?}", loss.as_slice().unwrap()); 
        
        graph.backward();
    
        for var_idx in graph.parameters() {
            let mut var = graph.node(var_idx);
            let grad = var.grad() * (lr / y.len() as f64);
            let delta = var.output() - grad;
            graph.mut_node_output(var_idx, delta.clone());
        }


    }

    println!("{:?}", graph.node(4)); 


}


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


    for epoch in 0..1000 {
        
        graph.forward();

        let loss_node = graph.curr_node();
        let loss = loss_node.output();
        println!("Loss: {:?}", loss.as_slice().unwrap()); 

        graph.backward();

        for var_idx in graph.parameters() {
            let mut var = graph.node(var_idx);
            let grad = var.grad() * (lr as f64);
            let delta = var.output() - grad;
            graph.mut_node_output(var_idx, delta.clone());
        }

    }

}



fn main() {



}
