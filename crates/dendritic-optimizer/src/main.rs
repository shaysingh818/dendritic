use std::io::Write; 

use chrono::Local; 
use ndarray::{arr2, Array2, Axis};

use dendritic_autodiff::operations::activation::*; 
use dendritic_autodiff::operations::loss::*; 
use dendritic_optimizer::classification::*; 
use dendritic_optimizer::regression::*;
use dendritic_optimizer::train::*;

pub fn load_data() -> (Array2<f64>, Array2<f64>) {

    let x = arr2(&[
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ]);

    let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

    (x, y)
}



fn main() -> std::io::Result<()> {

    env_logger::builder()
    .format(|buf, record| {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        writeln!(buf, "{}:{} {}", log_time, record.level(), record.args())
    }).init();



    // nesterov momentum
    let lr = 0.001; // learning rate
    let B = 0.9; // momentum factor
    let mut v_w = Array2::zeros((3, 1)); // velocity for weights
    let mut v_b = Array2::zeros((1,1)); // velocity for biases
    let (x, y) = load_data(); 
    
    let mut model = Regression::new(&x, &y, lr).unwrap();
    
    for _ in 0..250 {

        model.graph.forward(); 
        model.graph.backward();

        let mut w = model.graph.node(1); 
        let mut b = model.graph.node(3); 

        // factor in velocity
        let w_lookahead = w.output() - (B * v_w.clone()); 
        let b_lookahead = b.output() - (B * v_b.clone()); 

        model.graph.mut_node_output(1, w_lookahead);
        model.graph.mut_node_output(3, b_lookahead);

        let w_grad = w.grad() * lr; 
        let b_grad = (b.grad() * lr).sum_axis(Axis(0));

        v_w = w_grad + (B * v_w); 
        v_b = b_grad + (B * v_b);

        let new_w = w.output() - v_w.clone(); 
        let new_b = b.output() - v_b.clone();

        model.graph.mut_node_output(1, new_w); 
        model.graph.mut_node_output(3, new_b);

        let loss_total = model.measure_loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );

    }

    println!("{:?}", model.predicted()); 
    

    

    Ok(())

}
