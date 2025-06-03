use ndarray::prelude::*; 
use ndarray::{arr2, Array2}; 
use dendritic_autodiff::node::{Node};
use dendritic_autodiff::tensor::{Tensor}; 
use dendritic_autodiff::graph::*; 
use dendritic_autodiff::graph_interface::*;

use polars::prelude::*; 

fn write_df_to_file() -> Result<(), Box<dyn std::error::Error>>   {
    let s = Series::new("values".into(), &[1, 2, 3]);
    let df = DataFrame::new(vec![s.into()])?;
    //ParquetWriter::new("data.parquet").finish(&mut df.clone())?;
    Ok(())
}

fn main() {

    /*
    let s1 = Column::new("Fruit".into(), ["Apple", "Apple", "Pear"]);
    let s2 = Column::new("Color".into(), ["Red", "Yellow", "Green"]);

    let df: PolarsResult<DataFrame> = DataFrame::new(vec![s1, s2]);
    
    let lr: f64 = 0.02; 
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

    for epoch in 0..1000 {

        graph.forward();

        let loss_node = graph.node(6);
        let loss = loss_node.output();
        println!("Loss: {:?}", loss.as_slice().unwrap()); 
        
        graph.backward();

        let x = graph.node(0); 
        let mut w = graph.node(1); 
        let mut b = graph.node(3);

        let w_grad = w.grad() * (lr / y.len() as f64); 
        let dw = w.output() - w_grad;
        graph.mut_node_output(1, dw.clone());

        //let b_grad = b.grad().sum_axis(Axis(0)); 
        let db = b.grad() * (lr / y.len() as f64);
        graph.mut_node_output(3, db.clone());


    } */

    let x = arr2(&[
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]);

    let lr: f64 = 0.01; 
    let w = Array2::<f64>::zeros((2, 3));
    let b = Array2::<f64>::zeros((1, 3));
    let y = arr2(&[[0.0], [0.0], [1.0], [1.0]]); 

    let mut graph = ComputationGraph::new();
    graph.mul(vec![x, w]); 
    graph.add(vec![b]);
    graph.sigmoid();
    graph.mse(y.clone());

    for epoch in 0..1000 {

        graph.forward();

        let loss_node = graph.node(6);
        let loss = loss_node.output();
        println!("Loss: {:?}", loss.as_slice().unwrap()); 
        
        graph.backward();

        let x = graph.node(0); 
        let mut w = graph.node(1); 
        let mut b = graph.node(3);

        let w_grad = w.grad() * (lr / y.len() as f64); 
        let dw = w.output() - w_grad;
        graph.mut_node_output(1, dw.clone());

        //let b_grad = b.grad().sum_axis(Axis(0)); 
        let db = b.grad() * (lr / y.len() as f64);
        graph.mut_node_output(3, db.clone());

    } 

}
