use std::io::Write; 

use rand::thread_rng;
use rand::prelude::SliceRandom;
use chrono::Local; 
use ndarray::{s, arr2, Axis};

use dendritic_optimizer::descent::*; 
use dendritic_optimizer::regression::lasso::*;


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
    
    let mut model = LassoRegression::new(&x, &y, 0.001, 0.001).unwrap();
    model.train_batch(3000, 4); 

    

    //model.train_batch(10000, 5);

    /*
    let mut model = LinearRegression::load("testing").unwrap();
    let val = model.predict(&x);
    println!("{:?}", val); 
 
    let mut model_2 = LinearRegression::load("testing_2").unwrap();
    let val = model_2.predict(&x);
    println!("{:?}", val); */

    Ok(())

}
