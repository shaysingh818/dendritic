pub mod ndarray;
pub mod regression;
pub mod loss;
pub mod models;

// use crate::models::gtd::*;

use crate::regression::ridge::*;
use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*;
use crate::loss::rss::*;

fn main()  {



    let x: NDArray<f64> = NDArray::array(
        vec![5, 3], 
        vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0, 
            3.0, 4.0, 5.0,
            4.0, 5.0, 6.0, 
            5.0, 6.0, 7.0
        ]
    ).unwrap();

    let y: NDArray<f64> = NDArray::array(
        vec![5, 1], 
        vec![10.0, 12.0, 14.0, 16.0, 18.0]
    ).unwrap();

    let mut model = Ridge::new(x.clone(), y.clone(), 0.01).unwrap();
    model.train(1000, true);
    


}
