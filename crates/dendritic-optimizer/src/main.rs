use std::io::Write; 

use chrono::Local; 
use ndarray::{arr2, Array2, Axis};

use dendritic_autodiff::operations::activation::*; 
use dendritic_autodiff::operations::loss::*; 
use dendritic_optimizer::classification::*; 
use dendritic_optimizer::regression::*;
use dendritic_optimizer::train::*;


fn main() -> std::io::Result<()> {

    env_logger::builder()
    .format(|buf, record| {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        writeln!(buf, "{}:{} {}", log_time, record.level(), record.args())
    }).init();

    // binary logstic data
    let x = arr2(&[
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.8],
        [3.0, 3.2],
        [2.8, 3.0],
        [5.0, 5.5],
        [6.0, 5.8],
        [5.5, 6.0],
        [6.2, 5.9],
        [7.0, 6.5]
    ]);

    let y = arr2(&[
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0]
    ]);

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
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]);


    let mut model = Logistic::new(&x1, &y1, true, 0.01).unwrap();

    model.train(2000); 
    //println!("{:?}", model.graph.node(2)); 

    //model.train(1000);  
    let softmax = model.predicted();
    let output: Vec<_>  = softmax
        .axis_iter(Axis(0)) // iterate over rows
        .map(|row| {
            let (predicted_idx, &prob) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            (predicted_idx, prob)
        })
        .collect();

    println!("{:?}", output); 
    

    //model.train(1000); 




    Ok(())

}
