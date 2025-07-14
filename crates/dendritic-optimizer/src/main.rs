use std::io::Write; 

use chrono::Local; 
use ndarray::{arr2, Array2};

use dendritic_optimizer::regression::*;
use dendritic_optimizer::train::*;


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

    let mut model = Elastic::new(&x, &y, 0.001, 0.0001, 0.5).unwrap();
    model.train_batch(3000, 4);
    println!("{:?}", model.predict(&x)); 
    //model.save("linear_testing_2").unwrap();


    

    //model.train_batch(10000, 5);

    //let y_val: Array2<f64> = Array2::zeros((x.nrows(), 1));
    //let mut model = Regression::load("linear_testing_2").unwrap();
    //let val = model.predict(&x);
    //println!("{:?}", val); 


    Ok(())

}
