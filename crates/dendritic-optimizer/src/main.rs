use ndarray::{arr2}; 

use dendritic_optimizer::descent::*; 
use dendritic_optimizer::regression::*;


fn main() -> std::io::Result<()> {

     let x = arr2(&[
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ]);

    let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]); 

    //let mut model = LinearRegression::new(&x, &y, 0.008).unwrap();
    //model.train(1000);
    //model.save("testing_2")?;
    
    let mut model = LinearRegression::load("testing").unwrap();
    let val = model.predict(&x);
    println!("{:?}", val); 

 
    let mut model_2 = LinearRegression::load("testing_2").unwrap();
    let val = model_2.predict(&x);
    println!("{:?}", val); 

    Ok(())

}
