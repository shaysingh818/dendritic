use std::io::Write; 

use chrono::Local; 
use ndarray::{arr2, Array2, Axis};

use dendritic_autodiff::operations::activation::*; 
use dendritic_autodiff::operations::loss::*; 
use dendritic_optimizer::regression::logistic::*; 
use dendritic_optimizer::regression::sgd::*;
use dendritic_optimizer::regression::ridge::*;
use dendritic_optimizer::train::*;
use dendritic_optimizer::model::*;
use dendritic_optimizer::optimizers::*; 
use dendritic_optimizer::optimizers::Optimizer;

pub fn load_data() -> (Array2<f64>, Array2<f64>) {

    let x = arr2(&[
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ]);

    let y = arr2(&[
        [10.0], [12.0], [14.0], [16.0], [18.0],
        [10.0], [12.0], [14.0], [16.0], [18.0]
    ]);

    (x, y)
}

pub fn load_multi_class() -> (Array2<f64>, Array2<f64>, Array2<f64>) {

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

    let x2 = arr2(&[
        [4.5, 4.8],
        [5.0, 5.2],   // Class 1
        [7.0, 7.5],
        [7.5, 8.0],
        [8.0, 8.5],   // Class 2              
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 1.0],   // Class 0
        [4.0, 4.5],
        [4.5, 4.8],
        [5.0, 5.2],   // Class 1
        [7.0, 7.5],
        [7.5, 8.0],
        [8.0, 8.5]   // Class 2
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

    (x1, x2, y1)
}

pub fn load_binary_data() -> (Array2<f64>, Array2<f64>,  Array2<f64>) {

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

    let x2 = arr2(&[
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.8],
        [3.0, 3.2],
        [2.8, 3.0],
        [5.0, 5.5],
        [6.0, 5.8],
        [5.5, 6.0],
        [6.2, 5.9],
        [7.0, 6.5],
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

    (x, x2, y)
}


fn main() -> std::io::Result<()> {

    env_logger::builder()
    .format(|buf, record| {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        writeln!(buf, "{}:{} {}", log_time, record.level(), record.args())
    }).init(); 

    let (x, x1, y) = load_multi_class();

    let mut model = Logistic::new(&x, &y, true, 0.05).unwrap();
    model.train(1); 

    /*
    for _ in 0..1 {
        model.forward();
        model.backward();
        model.update_parameters(); 
        println!("Loss: {:?}", model.loss()); 
    } */

    

    let predicted = model.predict(&x1); 
    println!("{:?}", predicted); 

    Ok(())

}
