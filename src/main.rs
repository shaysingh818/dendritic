use ndarray::prelude::*; 
use ndarray::Array; 

fn main() {

    let a = array![
        [1, 2, 3],
        [4, 5, 6]
    ];

    println!("{:?}", a.shape());

    let b = Array::<f64, _>::zeros((3, 2, 4).f()); 
    println!("{:?}", b); 

}
