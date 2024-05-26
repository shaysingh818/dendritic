pub mod ndarray;
pub mod regression;
pub mod loss;
pub mod models;
pub mod autodiff; 

use crate::loss::mse::*;
use crate::ndarray::ndarray::NDArray;
use crate::ndarray::ops::*; 
use crate::autodiff::node::*;
use crate::autodiff::ops::*;

fn main()  {

    let x_path = "data/linear_testing_data/inputs";
    let y_path = "data/linear_testing_data/outputs"; 
    let w_path = "data/linear_testing_data/weights";
    let b_path = "data/linear_testing_data/bias";

    let x: NDArray<f64> = NDArray::load(x_path).unwrap();
    let y: NDArray<f64> = NDArray::load(y_path).unwrap();
    let w: NDArray<f64> = NDArray::load(w_path).unwrap();
    let b: NDArray<f64> = NDArray::load(b_path).unwrap();

    let inputs = Value::new(&x);
    let mut weights = Value::new(&w); // this needs to be mutable
    let mut biases = Value::new(&b);
    let learning_rate = 0.01; 

    let mut linear = ScaleAdd::new(
        Dot::new(inputs.clone(), weights.clone()),
        biases.clone()
    );

    for _epoch in 0..1000 {

        linear.forward();

        let y_pred = linear.value();
        let loss = mse(&y, &y_pred);
        let error = y_pred.subtract(y.clone()).unwrap();
        println!("Loss: {:?}", loss.unwrap()); 

        linear.backward(error);

        /* update weights */
        let learning_rate_factor = learning_rate/y_pred.size() as f64;
        let w_grad = inputs.grad().scalar_mult(learning_rate_factor).unwrap();
        let dw = weights.val().subtract(w_grad).unwrap();
        weights.set_val(&dw); 

        /* update biases */
        let b_collapse = biases.grad().sum_axis(1).unwrap();
        let db = b_collapse.scalar_mult(learning_rate_factor).unwrap();
        biases.set_val(&db);


    }

}
