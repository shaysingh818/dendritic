use std::io::Write; 

use chrono::Local; 
use ndarray::{s, arr2}; 

use dendritic_optimizer::descent::*; 
use dendritic_optimizer::regression::linear::*;


fn main() -> std::io::Result<()> {

    env_logger::builder()
    .format(|buf, record| {
        let now = Local::now(); 
        let log_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        writeln!(buf, "{}:{} {}", log_time, record.level(), record.args())
    }).init();
    

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


    let x_slice = x.slice(s![0..4, ..]); 
    let y_slice = y.slice(s![0..4, ..]);

    //println!("{:?}", x_slice); 

    let mut model = LinearRegression::new(&x, &y, 0.008).unwrap();

    /*
    model.function_definition(); 

    model.graph.mut_node_output(0, x_slice.to_owned());
    model.graph.node(0).set_grad_output(x_slice.to_owned()); 

    model.graph.mut_node_output(4, y_slice.to_owned()); 
    model.graph.mut_node_output(5, y_slice.to_owned());
    model.graph.node(5).set_grad_output(y_slice.to_owned()); 

    println!("{:?}", model.graph.nodes());

    model.graph.forward();
    model.graph.backward(); */


    model.train_batch(1000, 3);
    //model.save_snapshot("testing")?;

    /*
    let mut model = LinearRegression::load("testing").unwrap();
    let val = model.predict(&x);
    println!("{:?}", val); 
 
    let mut model_2 = LinearRegression::load("testing_2").unwrap();
    let val = model_2.predict(&x);
    println!("{:?}", val); */

    Ok(())

}
