pub mod ndarray;
pub mod regression;
pub mod loss;
pub mod models;
pub mod autodiff; 

use crate::loss::mse::*;
use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*;
use crate::autodiff::ops::*;
use crate::autodiff::node::*;


fn main()  {


    let x: NDArray<f64> = NDArray::load("../../data/linear_testing_data/inputs").unwrap();
    let y: NDArray<f64> = NDArray::load("../../data/linear_testing_data/outputs").unwrap();
    let w: NDArray<f64> = NDArray::load("../../data/linear_testing_data/weights").unwrap();
    let b: NDArray<f64> = NDArray::load("../../data/linear_testing_data/bias").unwrap();

    let inputs = Value::new(&x);
    let weights = Value::new(&w); // this needs to be mutable
    let _learning_rate = 0.01; 

    let mut linear = ScaleAdd::new(
      Dot::new(inputs.clone(), weights.clone()),
      Value::new(&b)
    );

    for _epoch in 0..1 {

        linear.forward();

        let y_pred = linear.value();
        let loss = mse(&y, &y_pred);
        println!("Loss: {:?}", loss.unwrap());

        let error = y.subtract(y_pred.clone()).unwrap();    

        linear.backward(error);

        /* update weights */
        let w_grad = weights.grad();
        let b_grad = linear.lhs().value();

        println!("{:?}", w_grad);
        println!("{:?}", b_grad);

        // let dw = weights.grad().scalar_mult(learning_rate).unwrap();
        // let grad = learning_rate/y_pred.size() as f64; 
        // let db: f64 = 

        // let error = y_pred.subtract(self.outputs.clone()).unwrap();
        // let grad = self.learning_rate/y_pred.size() as f64;
        // let db: f64 = error.scalar_mult(grad).unwrap().values().iter().sum();
        // let dw = weights.value().scalar_mult(learning_rate).unwrap();
        // let new_weights = weights.value().subtract(dw).unwrap(); 

    }

    // println!("{:?}", inputs.gradient); 
    // println!("{:?}", weights.gradient);

}
